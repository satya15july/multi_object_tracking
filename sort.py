#!\bin\python2.7

"""
Main module for the real-time tracker class execution. Based on the SORT algorithm
"""

from __future__ import print_function
import os.path
from tracker import KalmanTracker, ORBTracker, ReIDTracker
import numpy as np
import cv2

import random
import colorsys
from object_detection_api import ObjectDetectorAPI

from netutils import ObjDetArchType
import config

class SORT:

    def __init__(self, src=None, tracker='Kalman', detector='dyhead-fpn', benchmark=False):
        """
         Sets key parameters for SORT
        :param src: path to video file
        :param tracker: (string) 'ORB', 'Kalman' or 'ReID', determines which Tracker class will be used for tracking
        :param benchmark: (bool) determines whether the track will perform a test on the MOT benchmark

        ---- attributes ---
        detections (list) - relevant for 'benchmark' mode, data structure for holding all the detections from file
        frame_count (int) - relevant for 'benchmark' mode, frame counter, used for indexing and looping through frames
        """
        if tracker == 'Kalman': self.tracker = KalmanTracker(metric='hybrid')
        elif tracker == 'ORB': self.tracker = ORBTracker()

        self.benchmark = benchmark
        if src is not None:
            self.src = cv2.VideoCapture(src)

        width = int(self.src.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.src.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = self.src.get(cv2.CAP_PROP_FPS)
        #num_frames = int(self.src.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.src is not None:
            self.video_writer = cv2.VideoWriter(config.VIDEO_OUTPUT_FILE_NAME, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                           fps=float(frames_per_second),
                                           frameSize=(width, height), isColor=True)

        print("width: {}, height: {}".format(width, height))
        self.tracker.set_width_height(width, height)

        self.detector = None

        if self.benchmark:
            SORT.check_data_path()
            self.sequences = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof']
            """
            More sequences:
            'ETH-Sunnyday', 'ETH-Pedcross2', 'KITTI-13', 'KITTI-17', 'ADL-Rundle-6', 'ADL-Rundle-8', 'Venice-2'
            """
            self.seq_idx = None
            self.load_next_seq()

        else:
            if detector == 'dyhead-fpn':
                self.detector = ObjectDetectorAPI(ObjDetArchType.DYHEAD_FPN)
            self.score_threshold = 0.7
            self.start_tracking()
    def close(self):
        self.src.release()
        self.video_writer.release()

    def load_next_seq(self):
        """
        When switching sequence - propagate the sequence index and reset the frame count
        Load pre-made detections for .txt file (from MOT benchmark). Starts tracking on next sequence
        """
        if self.seq_idx == len(self.sequences) - 1:
            print('SORT finished going over all the input sequences... closing tracker')
            return

        # Load detection from next sequence and reset the frame count for it
        if self.seq_idx is None:
            self.seq_idx = 0
        else:
            self.seq_idx += 1
        self.frame_count = 1

        # Load detections for new sequence
        file_path = 'data/%s/det.txt' % self.sequences[self.seq_idx]
        self.detections = np.loadtxt(file_path, delimiter=',')

        # reset the tracker and start tracking on new sequence
        self.tracker.reset()
        self.start_tracking()

    def next_frame(self):
        """
        Method for handling the correct way to fetch the next frame according to the 'src' or
         'benchmark' attribute applied
        :return: (np.ndarray) next frame, (np.ndarray) detections for that frame
        """
        if self.benchmark:
            frame = SORT.show_source(self.sequences[self.seq_idx], self.frame_count)
            new_detections = self.detections[self.detections[:, 0] == self.frame_count, 2:7]
            new_detections[:, 2:4] += new_detections[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            self.frame_count += 1
            return frame, new_detections[:, :4]

        else:
            _, frame = self.src.read()
            #frame = cv2.resize(frame, (1280, 720))
            print("frame{}".format(frame.shape))
            boxes, scores, classes, num = self.detector.processFrame(frame)
            print("boxes: {}".format(boxes))

            return frame, boxes
    def start_tracking(self):
        while True:
            # Fetch the next frame from video source, if no frames are fetched, stop loop
            frame, detections = self.next_frame()
            print("tracking main loop, frame:{}, detections:{}".format(frame, detections))
            if frame is None:
                break

            # Send new detections to set tracker
            if isinstance(self.tracker, KalmanTracker):
                tracks = self.tracker.update(detections)
            elif isinstance(self.tracker, ORBTracker) or isinstance(self.tracker, ReIDTracker):
                tracks = self.tracker.update(frame, detections)
            else:
                raise Exception('[ERROR] Tracker type not specified for SORT')

            # Look through each track and display it on frame (each track is a tuple (ID, [x1,y1,x2,y2])
            for ID, bbox in tracks:
                print("tracking for loop, bbox:{}".format(bbox))
                #bbox = self.verify_bbox_format(bbox)
                # Generate pseudo-random colors for bounding boxes for each unique ID
                random.seed(ID)

                # Make sure the colors are strong and bright and draw the bounding box around the track
                h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
                color = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]

                x1, y1, x2, y2 = [int(i) for i in bbox]
                # cv2.rectangle(img,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(255,0,0),2)
                color = config.COLORS_150[ID % len(config.COLORS_150)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                t_size = cv2.getTextSize(str(ID), cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + t_size[0] + 3, (y1 - 20) + t_size[1] + 4), color,
                                  -1)
                cv2.putText(frame, str(ID), (x1, (y1 - 20) + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2,
                                [255, 255, 255], 2)

            # Show tracked frame
            cv2.imshow("Video Feed", frame)
            self.video_writer.write(frame)

            # if the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print('SORT operation terminated by user... closing tracker')
                return

        if self.benchmark:
            self.load_next_seq()

    def verify_bbox_format(self, bbox):
        """
        Fixes bounding box format according to video type (e.g. benchmark test or video capture)
        :param bbox: (array) list of bounding boxes
        :return: (array) reformatted bounding box
        """
        if self.benchmark:
            return bbox.astype("int")
        else:
            bbox.astype("int")
            return [bbox[1], bbox[0], bbox[3], bbox[2]]


    @staticmethod
    def show_source(seq, frame, phase='train'):
        """ Method for displaying the origin video being tracked """
        return cv2.imread('mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame))

    @staticmethod
    def check_data_path():
        """ Validates correct implementation of symbolic link to data for SORT """
        if not os.path.exists('mot_benchmark'):
            print('''
            ERROR: mot_benchmark link not found!\n
            Create a symbolic link to the MOT benchmark\n
            (https://motchallenge.net/data/2D_MOT_2015/#download)
            ''')
            exit()


def main():
    """ Starts the tracker on source video. Can start multiple instances of SORT in parallel """
    path_to_video = config.VIDEO_INPUT_FILE_NAME
    print("path_to_video: {}".format(path_to_video))
    mot_tracker = SORT(path_to_video)


if __name__ == '__main__':
    main()
