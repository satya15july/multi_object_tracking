
DATASET_TRAIN = 'cityscapes_fine_instance_seg_train'
DATASET_VAL = 'cityscapes_fine_instance_seg_val'
MODEL_WEIGHT_PATH = 'out_model/object_det_models/dyhead_fpn/model_final.pth'
TARGET_DEVICE = 'cuda'

CITYSCAPES_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
OBJ_DET_MODEL_OUTDIR = 'out_model/object_det_models'
VIDEO_OUTPUT_PATH='video_output'

# Use input/video/nvidia.mp4, cityscapes_long.mp4, pexels_.mp4, mot17_1.mp4, car_video.mp4
VIDEO_INPUT_FILE_NAME = "input/video/mot17_2.mp4"
VIDEO_OUTPUT_FILE_NAME = 'video_output/sort_result.mp4'

FONT_SCALE = 2e-3
THICKNESS_SCALE = 1e-3

DEBUG_FLAG = True

FRAME_WIDTH = 512
FRAME_HEIGHT = 1024

COLORS_150 = [(222, 253, 60), (222, 253, 212), (222, 253, 37), (222, 253, 210), (122, 253, 141), (222, 253, 149),
             (222, 253, 81), (222, 253, 250), (222, 253, 79), (222, 253, 111), (222, 39, 60), (222, 39, 212),
             (222, 39, 37), (222, 39, 210), (222, 39, 141), (222, 39, 149), (222, 39, 81), (222, 39, 250),
             (222, 39, 79), (222, 39, 111), (222, 93, 60), (222, 93, 212), (222, 93, 37), (222, 93, 210),
             (222, 93, 141), (222, 93, 149), (222, 93, 81), (222, 93, 250), (222, 93, 79), (222, 93, 111),
             (222, 238, 60), (222, 238, 212), (222, 238, 37), (222, 238, 210), (222, 238, 141), (222, 238, 149),
             (222, 238, 81), (222, 238, 250), (222, 238, 79), (222, 238, 111), (222, 13, 60), (222, 13, 212),
             (222, 13, 37), (222, 13, 210), (222, 13, 141), (222, 13, 149), (222, 13, 81), (222, 13, 250),
             (222, 13, 79), (222, 13, 111), (222, 184, 60), (222, 184, 212), (222, 184, 37), (222, 184, 210),
             (222, 184, 141), (222, 184, 149), (222, 184, 81), (222, 184, 250), (222, 184, 79), (222, 184, 111),
             (222, 146, 60), (222, 146, 212), (222, 146, 37), (222, 146, 210), (222, 146, 141), (222, 146, 149),
             (222, 146, 81), (222, 146, 250), (222, 146, 79), (222, 146, 111), (222, 22, 60), (222, 22, 212),
             (222, 22, 37), (222, 22, 210), (222, 22, 141), (222, 22, 149), (222, 22, 81), (222, 22, 250),
             (222, 22, 79), (222, 22, 111), (222, 204, 60), (222, 204, 212), (222, 204, 37), (222, 204, 210),
             (222, 204, 141), (222, 204, 149), (222, 204, 81), (222, 204, 250), (222, 204, 79), (222, 204, 111),
             (222, 151, 60), (222, 151, 212), (222, 151, 37), (222, 151, 210), (222, 151, 141), (222, 151, 149),
             (222, 151, 81), (222, 151, 250), (222, 151, 79), (222, 151, 111), (53, 253, 60), (53, 253, 212),
             (53, 253, 37), (53, 253, 210), (53, 253, 141), (53, 253, 149), (53, 253, 81), (53, 253, 250),
             (53, 253, 79), (53, 253, 111), (53, 39, 60), (53, 39, 212), (53, 39, 37), (53, 39, 210), (53, 39, 141),
             (53, 39, 149), (53, 39, 81), (53, 39, 250), (53, 39, 79), (53, 39, 111), (53, 93, 60), (53, 93, 212),
             (53, 93, 37), (53, 93, 210), (53, 93, 141), (53, 93, 149), (53, 93, 81), (53, 93, 250), (53, 93, 79),
             (53, 93, 111), (53, 238, 60), (53, 238, 212), (53, 238, 37), (53, 238, 210), (53, 238, 141),
             (53, 238, 149), (53, 238, 81), (53, 238, 250), (53, 238, 79), (53, 238, 111), (53, 13, 60),
             (53, 13, 212), (53, 13, 37), (53, 13, 210), (53, 13, 141), (53, 13, 149), (53, 13, 81),
             (53, 13, 250), (53, 13, 79), (53, 13, 111)]
