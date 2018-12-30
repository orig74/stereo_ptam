import numpy as np

import time
from itertools import chain
from collections import defaultdict

from covisibility import CovisibilityGraph
from optimization import BundleAdjustment
from mapping import Mapping
from mapping import MappingThread
from components import Measurement
from motion import MotionModel
from loopclosing import LoopClosing

from params import ParamsBlueROV

import sys, traceback
import zmq
import pickle

context = zmq.Context()

def subscribe(topics,port,ip='127.0.0.1'):
    zmq_sub = context.socket(zmq.SUB)
    zmq_sub.connect("tcp://%s:%d" % (ip,port))
    for topic in topics:
        zmq_sub.setsockopt(zmq.SUBSCRIBE,topic)
    return zmq_sub

zmq_sub = subscribe([b'viewer_pub_image_topic'],10921)

class Tracking(object):
    def __init__(self, params):
        self.optimizer = BundleAdjustment()
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations

    def refine_pose(self, pose, cam, measurements):
        assert len(measurements) >= self.min_measurements, (
            'Not enough points')

        self.optimizer.clear()
        self.optimizer.add_pose(0, pose, cam, fixed=False)

        for i, m in enumerate(measurements):
            self.optimizer.add_point(i, m.mappoint.position, fixed=True)
            self.optimizer.add_edge(0, i, 0, m)

        self.optimizer.optimize(self.max_iterations)
        return self.optimizer.get_pose(0)



class SPTAM(object):
    def __init__(self, params):
        self.params = params



        self.loop_closing = None# LoopClosing(self, params)
        self.loop_correction = None

        self.reference = None        # reference keyframe
        self.preceding = None        # last keyframe
        self.current = None          # current frame
        self.status = defaultdict(bool)
        self.graph = CovisibilityGraph()
        self.mapping = MappingThread(self.graph, self.params)
        self.tracker = Tracking(self.params)

    def stop(self):
        self.mapping.stop()
        if self.loop_closing is not None:
            self.loop_closing.stop()

    def initialize(self, frame, initial_position=True):
        if initial_position:
            self.motion_model = MotionModel()
        else:
            pose, _ = self.motion_model.predict_pose(frame.timestamp)
            #import pdb;pdb.set_trace()
            frame.update_pose(pose)
            #frame.left,frame.right=frame.right,frame.left
        mappoints, measurements = frame.triangulate()
        if len(mappoints) < self.params.init_min_points:
             print('Not enough points to initialize map.',len(mappoints),len(measurements))
             return False

        keyframe = frame.to_keyframe()
        keyframe.set_fixed(True)
        self.graph.add_keyframe(keyframe)
        self.mapping.add_measurements(keyframe, mappoints, measurements)
        if self.loop_closing is not None:
            self.loop_closing.add_keyframe(keyframe)

        self.reference = keyframe
        self.preceding = keyframe
        self.current = keyframe
        self.status['initialized'] = True

        self.motion_model.update_pose(
            frame.timestamp, frame.position, frame.orientation)
        return True


    def track(self, frame):
        while self.is_paused():
            time.sleep(1e-4)
        self.set_tracking(True)

        self.current = frame
        print('Tracking:', frame.idx, ' <- ', self.reference.id, self.reference.idx)

        predicted_pose, _ = self.motion_model.predict_pose(frame.timestamp)
        frame.update_pose(predicted_pose)

        if self.loop_closing is not None:
            if self.loop_correction is not None:
                estimated_pose = g2o.Isometry3d(
                    frame.orientation,
                    frame.position)
                estimated_pose = estimated_pose * self.loop_correction
                frame.update_pose(estimated_pose)
                self.motion_model.apply_correction(self.loop_correction)
                self.loop_correction = None

        local_mappoints = self.filter_points(frame)
        measurements = frame.match_mappoints(
            local_mappoints, Measurement.Source.TRACKING)

        print('measurements:', len(measurements), '   ', len(local_mappoints))

        tracked_map = set()
        for m in measurements:
            mappoint = m.mappoint
            mappoint.update_descriptor(m.get_descriptor())
            mappoint.increase_measurement_count()
            tracked_map.add(mappoint)

        try:
            if len(measurements)<=self.tracker.min_measurements:
                self.initialize(frame,False)
            else:
                self.reference = self.graph.get_reference_frame(tracked_map)
                pose = self.tracker.refine_pose(frame.pose, frame.cam, measurements)
                frame.update_pose(pose)
                self.motion_model.update_pose(
                    frame.timestamp, pose.position(), pose.orientation())
            tracking_is_ok = True
        except:
            tracking_is_ok = False
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            print('tracking failed!!!')
            return False
            #self.initialize(frame)

        if tracking_is_ok and self.should_be_keyframe(frame, measurements):
            print('new keyframe', frame.idx)
            keyframe = frame.to_keyframe()
            keyframe.update_reference(self.reference)
            keyframe.update_preceding(self.preceding)
            try:
                self.mapping.add_keyframe(keyframe, measurements)
                if self.loop_closing is not None:
                    self.loop_closing.add_keyframe(keyframe)
                self.preceding = keyframe
            except:
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                #self.initialize(frame)
                print('failed to add keyframe')
                return False

        self.set_tracking(False)
        return True


    def filter_points(self, frame):
        local_mappoints = self.graph.get_local_map_v2(
            [self.preceding, self.reference])[0]

        can_view = frame.can_view(local_mappoints)
        print('filter points:', len(local_mappoints), can_view.sum(),
            len(self.preceding.mappoints()),
            len(self.reference.mappoints()))

        checked = set()
        filtered = []
        for i in np.where(can_view)[0]:
            pt = local_mappoints[i]
            if pt.is_bad():
                continue
            pt.increase_projection_count()
            filtered.append(pt)
            checked.add(pt)

        for reference in set([self.preceding, self.reference]):
            for pt in reference.mappoints():  # neglect can_view test
                if pt in checked or pt.is_bad():
                    continue
                pt.increase_projection_count()
                filtered.append(pt)

        return filtered


    def should_be_keyframe(self, frame, measurements):
        if self.adding_keyframes_stopped():
            return False

        n_matches = len(measurements)
        n_matches_ref = len(self.reference.measurements())

        print('keyframe check:', n_matches, '   ', n_matches_ref)

        return ((n_matches / n_matches_ref) <
            self.params.min_tracked_points_ratio) or n_matches < 20


    def set_loop_correction(self, T):
        self.loop_correction = T

    def is_initialized(self):
        return self.status['initialized']

    def pause(self):
        self.status['paused'] = True

    def unpause(self):
        self.status['paused'] = False

    def is_paused(self):
        return self.status['paused']

    def is_tracking(self):
        return self.status['tracking']

    def set_tracking(self, status):
        self.status['tracking'] = status

    def stop_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = True

    def resume_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = False

    def adding_keyframes_stopped(self):
        return self.status['adding_keyframes_stopped']


""" for blue rov stereo rig
Intrinsic_mtx_1 [[505.67822359   0.         321.97576615]
 [  0.         506.12264948 258.28130924]
 [  0.           0.           1.        ]]
dist_1 [[-2.66779903e-01  1.23830866e-01  4.84788961e-05  3.01897996e-04
   2.73538545e-02]]
Intrinsic_mtx_2 [[508.52126077   0.         323.63553745]
 [  0.         508.87917314 266.09712194]
 [  0.           0.           1.        ]]
dist_2 [[-0.2811816   0.19784149  0.0024773   0.0011536  -0.07891902]]
R [[ 9.99997281e-01 -7.39107393e-04 -2.21175550e-03]
 [ 7.50148997e-04  9.99987241e-01  4.99557695e-03]
 [ 2.20803501e-03 -4.99722252e-03  9.99985076e-01]]
T [[-0.12211419]
 [-0.00052587]
 [-0.00208841]]
E [[ 4.05479154e-07  2.09100733e-03 -5.15428095e-04]
 [-1.81876800e-03 -6.08688248e-04  1.22116992e-01]
 [ 4.34263466e-04 -1.22113026e-01 -6.11193951e-04]]
F [[-2.48575687e-09 -1.28086169e-05  4.96141226e-03]
 [ 1.11409951e-05  3.72561660e-06 -3.83902297e-01]
 [-4.27025578e-03  3.82474663e-01  1.00000000e+00]]
ret =  0.7917227535575992
took 1.750295877456665
"""

class dataset:
    _fov=60.97
    _pixelwidthx = 640 #after shrink
    _pixelwidthy = 512 #after shrink
    _baseline = 0.122 # (240-100)*.1scale in cm from unreal engine
    _focal_length=_pixelwidthx/( np.tan(np.deg2rad(_fov/2)) *2 )
    class cam:
        pass
    cam.fx=_focal_length
    cam.fy=_focal_length
    cam.cx=_pixelwidthx//2
    cam.cy=_pixelwidthy//2
    cam.width=_pixelwidthx
    cam.height=_pixelwidthy
    #cam.baseline=0.14 #for sim
    cam.baseline=_baseline #for real


if __name__ == '__main__':
    import cv2
    import g2o

    import os
    import sys
    import argparse

    from threading import Thread

    from components import Camera
    from components import StereoFrame
    from feature import ImageFeature
    #from params import ParamsKITTI, ParamsEuroc
    from dataset import KITTIOdometry, EuRoCDataset


    parser = argparse.ArgumentParser()
    parser.add_argument('--no-viz', action='store_true', help='do not visualize')
    parser.add_argument('--dataset', type=str, help='dataset (KITTI/EuRoC)',
        default='KITTI')
    #parser.add_argument('--path', type=str, help='dataset path',
    #    default='path/to/your/KITTI_odometry/sequences/00')
    args = parser.parse_args()

    params = ParamsBlueROV()

    sptam = SPTAM(params)

    visualize = not args.no_viz
    if visualize:
        from viewer import MapViewer
        viewer = MapViewer(sptam, params)


    cam = Camera(
        dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy,
        dataset.cam.width, dataset.cam.height,
        params.frustum_near, params.frustum_far,
        dataset.cam.baseline)



    durations = []
    timestamp=0
    i=0

    while 1:
        if len(zmq.select([zmq_sub],[],[],0.001)[0])>0:
            ret = zmq_sub.recv_multipart()
            topic , data = ret
            data=pickle.loads(ret[1])

            featurel = ImageFeature(data[0], params)
            featurer = ImageFeature(data[1], params)

            time_start = time.time()

            t = Thread(target=featurer.extract)
            t.start()
            featurel.extract()
            t.join()

            frame = StereoFrame(i, g2o.Isometry3d(), featurel, featurer, cam, timestamp=timestamp)

            tret=False
            if not sptam.is_initialized():
                try:
                    sptam.initialize(frame)
                except:
                    print("-"*60)
                    traceback.print_exc(file=sys.stdout)
                    print("-"*60)
                    print('initialize failed!!!')
                print('initialize!!!')
            else:
                try:
                    tret=sptam.track(frame)
                except:
                    print("-"*60)
                    traceback.print_exc(file=sys.stdout)
                    print("-"*60)
                    print('tracking failed!!!')

            #if not tret:
        #        try:
        #            sptam.initialize(frame,False)
        #        except:
        #            print("-"*60)
        #            traceback.print_exc(file=sys.stdout)
        #            print("-"*60)
        #            print('initialize recover failed!!!')
                    #timestamp=-0.1
                    #i=-1

        #        print('initialize recover!!!')


            duration = time.time() - time_start
            durations.append(duration)
            print('duration', duration)
            print()
            print()

            if tret and visualize:
                try:
                    viewer.update()
                except:
                    print('Exception: fail to update viewer')

            timestamp+=0.1
            i+=1

    print('num frames', len(durations))
    print('num keyframes', len(sptam.graph.keyframes()))
    print('average time', np.mean(durations))


    sptam.stop()
    if visualize:
        viewer.stop()
