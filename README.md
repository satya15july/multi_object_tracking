# Multi-Object Tracking

https://user-images.githubusercontent.com/22910010/207591804-0767672c-89b3-465a-90c3-741c7dd85c6c.mp4

## Dependencies:
Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) by following the instruction given in their website .

(Note: python3.8 and above is required for detectron2.)

## Enviornment Setup:
-   Setting up external repos
    -   Create a folder name "ext_detectron2_net" from the root and clone [DyHeadRepo](https://github.com/microsoft/DynamicHead).
    -   Inside "ext_detectron2_net" folder,get the clone of [DETR](https://github.com/facebookresearch/detr)

- Download Pre-trained model for object detection.
    I have trained the DyHeadFPN model.Please download it from [Download Link](https://drive.google.com/drive/folders/1nzIahVtfG_fC4GqgqgUqLkUuw__S9_SE?usp=share_link).

    Keep the *.pth file inside 'out_model/object_del_models/dyhead_fpn'

# How to Run:

```
python3.8 demo.py

```


---
Reach me @

[LinkedIn](https://www.linkedin.com/in/satya1507/) [GitHub](https://github.com/satya15july) [Medium](https://medium.com/@satya15july_11937)

