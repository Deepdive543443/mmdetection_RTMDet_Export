#!/bin/bash

mim download mmdet --config rtmdet_x_8xb32-300e_coco --dest .
python demo/rtmdet_export_demo.py demo/demo.jpg rtmdet_x_8xb32-300e_coco.py --weights rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth --device cpu
