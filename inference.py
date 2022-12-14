# Usage: python3.8 inference_cityscapes.py --arch centermask_mv2 \
#                  --model model_out/centermasklite_mv2/model_final.pth --target cpu --inference image --save 0

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

import config

setup_logger()

# import some common libraries
import cv2, random

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode

from netutils import InstanceNetArch, ArchType

import argparse
import time
import tqdm
import os

from netutils import ObjectDetectionArch, ObjDetArchType

ap = argparse.ArgumentParser()
ap.add_argument('-a', '--arch', default='detr', choices=['detr', 'dyhead_fpn', 'dyhead_swint'], help='Choose different transformer based object detection architecture')
ap.add_argument("-m", "--model", required=True,	help="path  of the model")
ap.add_argument('-t', '--target', default='cpu', choices=['cpu', 'cuda'], help='Choose the target device')
ap.add_argument('-i', '--inference', default='image', choices=['image', 'webcam', 'video_input'], help='Choose the inference type')
ap.add_argument("-f", "--file", required=False,	help="path  of the video/image file")
ap.add_argument("-s", '--save', default=0, type = int, help='save predicted output')
args = vars(ap.parse_args())

ARCHITECTURE = args['arch']
print("ARCHITECTURE: {} ".format(ARCHITECTURE))

DATASET_TRAIN = 'cityscapes_fine_instance_seg_train'
DATASET_VAL = 'cityscapes_fine_instance_seg_val'

CLASSES = config.CITYSCAPES_CLASSES
cityscapes_metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_train")
dataset_dicts = DatasetCatalog.get("cityscapes_fine_instance_seg_train")

model_weight = args['model']
target_device = args['target']
print("model_weight: {}, target_device: {}".format(model_weight, target_device))

arch_type = None
if args['arch'] == 'detr':
    arch_type = ObjDetArchType.DETR
elif args['arch'] == 'dyhead_fpn':
    arch_type = ObjDetArchType.DYHEAD_FPN
elif args['arch'] == 'dyhead_swint':
    arch_type = ObjDetArchType.DYHEAD_SWINT

object_detection = ObjectDetectionArch(len(CLASSES), arch_type)
object_detection.register_dataset(DATASET_TRAIN, DATASET_VAL)
object_detection.set_model_weights(model_weight)
object_detection.set_target_device(args['target'])
object_detection.set_score_threshold(0.7)
object_detection.set_confidence_threhold(0.7)

object_detection.print_cfg()

predictor = object_detection.default_predictor()

def run_demo_on_val_dataset():
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        print("image shape {}".format(im.shape))
        print("====PREDICTION======= STARTS")
        start = time.time()
        outputs = predictor(im)
        end = time.time()
        elapsed_time = (end - start) * 1000
        print("Evaluation Time for arch: {} on device: {} is {} ms ".format(args['arch'], target_device, elapsed_time))
        print("====PREDICTION======= ENDS")
        v = Visualizer(im[:, :, ::-1],
                       metadata=cityscapes_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(im[:, :, ::-1])  # BGR to RGB
        ax[0].set_title('Original Image ')
        ax[1].imshow(out.get_image()[:, :, ::-1])  # BGR to RGB
        ax[1].set_title('Segmented image ')
        #plt.show()
        if args['save']:
            filename = "output_images/output_{}.png".format(args['arch'])
            plt.savefig(filename, dpi=100)
        else:
            plt.show()

def filter_predictions_from_outputs(outputs,
                                    threshold=0.7,
                                    verbose=False):
    predictions = outputs["instances"].to("cpu")
    if verbose:
        print(list(predictions.get_fields()))
    indices = [i
               for (i, s) in enumerate(predictions.scores)
               if s >= threshold
               ]

    filtered_predictions = predictions[indices]

    return filtered_predictions

def run_demo_on_image():
    img_name1 = 'input_images/frankfurt_000000_003357_leftImg8bit.png'
    im = cv2.imread(img_name1)

    print("image shape {}".format(im.shape))
    print("====PREDICTION======= STARTS")
    start = time.time()
    outputs = predictor(im)
    end = time.time()
    elapsed_time = (end - start) * 1000
    print("Evaluation Time for arch: {} on device: {} is {} ms ".format(args['arch'], target_device, elapsed_time))
    #print('outputs {}'.format(outputs))
    filter_outputs=filter_predictions_from_outputs(outputs, threshold=0.5)
    #print('filter_outputs {}'.format(filter_outputs))
    print("====PREDICTION======= ENDS")
    v = Visualizer(im[:, :, ::-1],
                   metadata=dataset_dicts,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
    #out = v.draw_instance_predictions(filter_outputs["instances"].to("cpu"))
    out = v.draw_instance_predictions(filter_outputs)
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(im[:, :, ::-1])  # BGR to RGB
    ax[0].set_title('Original Image ')
    ax[1].imshow(out.get_image()[:, :, ::-1])  # BGR to RGB
    ax[1].set_title('Segmented image ')
    # plt.show()
    if args['save']:
        filename = "output_images/output_{}.png".format(args['arch'])
        plt.savefig(filename, dpi=100)
    else:
        plt.show()

def run_demo_on_webcam(object_detection):
    object_detection.run_on_webcam()

def run_demo_on_video_input(instance_seg, video_input):
    output = config.VIDEO_OUTPUT_PATH
    os.makedirs(output, exist_ok=True)
    instance_seg.run_on_video_input(video_input, output)

if args['inference'] == 'image':
    #run_demo_on_image()
    run_demo_on_val_dataset()
elif args['inference'] == 'webcam':
    run_demo_on_webcam(object_detection)
elif args['inference'] == 'video_input':
    #video_file = 'input_images/out_cityscapes.mp4'
    video_file = args['file']
    run_demo_on_video_input(object_detection, video_file)