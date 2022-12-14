# python3.8 training_cityscapes.py --arch centermask_mv2 --path <model_out> --epochs <> --model<> --resume<0/1>
# For example,

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, cv2, random

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
import argparse
import config

from netutils import ObjectDetectionArch, ObjDetArchType

ap = argparse.ArgumentParser()
ap.add_argument('-a', '--arch', default='detr', choices=['detr', 'dyhead_fpn', 'dyhead_swint'], help='Choose different transformer based object detection architecture')
#ap.add_argument("-p", "--path", required=True,	help="output path  to the model")
ap.add_argument("-e", "--epochs", type=int, help="No of Epochs for training")
ap.add_argument("-m", "--model", required=False,	help="Pre-trained model weight required for resume")
ap.add_argument("-r", '--resume', default=0, type=int, help='resume the training')
args = vars(ap.parse_args())

DATASET_TRAIN = 'cityscapes_fine_instance_seg_train'
DATASET_VAL = 'cityscapes_fine_instance_seg_train'

CLASSES = config.CITYSCAPES_CLASSES
cityscapes_metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_train")
dataset_dicts = DatasetCatalog.get("cityscapes_fine_instance_seg_train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=cityscapes_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(img[:, :, ::-1])  # BGR to RGB
    ax[0].set_title('Original Image ')
    ax[1].imshow(out.get_image()[:, :, ::-1])  # BGR to RGB
    ax[1].set_title('Segmented Image ')
    plt.show()

arch_type = None
if args['arch'] == 'detr':
    arch_type = ObjDetArchType.DETR
elif args['arch'] == 'dyhead_fpn':
    arch_type = ObjDetArchType.DYHEAD_FPN
elif args['arch'] == 'dyhead_swint':
    arch_type = ObjDetArchType.DYHEAD_SWINT

#path = args['path']
#model_output_path = os.path.join(path, args['arch'])
base_path = config.OBJ_DET_MODEL_OUTDIR
model_output_path = os.path.join(base_path, args['arch'])
print("model_output_path {}".format(model_output_path))
os.makedirs(model_output_path, exist_ok=True)

pre_trained_weight = args['model']
if args['resume']:
    resume_flag = True
else:
    resume_flag = False

print("pre_trained_weight {} ".format(pre_trained_weight))
print("resume_flag {} ".format(resume_flag))

object_detection = ObjectDetectionArch(len(CLASSES), arch_type)
object_detection.set_model_output_path(model_output_path)
object_detection.register_dataset(DATASET_TRAIN, DATASET_VAL)
object_detection.print_cfg()
object_detection.set_epochs(args['epochs'])

if resume_flag:
    object_detection.set_model_weights(pre_trained_weight)
    object_detection.train(resume_flag)
else:
    object_detection.train()
