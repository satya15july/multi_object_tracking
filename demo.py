from sort import SORT
import config
import argparse

"""
ap = argparse.ArgumentParser()
ap.add_argument('-a', '--detection', default='dyhead-fpn', choices=['dyhead-fpn', 'detr'], help='Choose Object Detection architecture')
ap.add_argument('-t', '--target', default='cuda', choices=['cpu', 'cuda'], help='Choose the target device')
ap.add_argument("-s", '--save', default=0, type = int, help='save predicted output')
args = vars(ap.parse_args())
"""

path_to_video = config.VIDEO_INPUT_FILE_NAME
print("path_to_video: {}".format(path_to_video))
mot_tracker = SORT(path_to_video)
