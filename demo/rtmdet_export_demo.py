# Copyright (c) OpenMMLab. All rights reserved.
"""Image Demo.

This script adopts a new infenence class, currently supports image path,
np.array and folder input formats, and will support video and webcam
in the future.

Example:
    Save visualizations and predictions results::

        python demo/image_demo.py demo/demo.jpg rtmdet-s

        python demo/image_demo.py demo/demo.jpg \
        configs/rtmdet/rtmdet_s_8xb32-300e_coco.py \
        --weights rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 --texts bench

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 --texts 'bench . car .'

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365
        --texts 'bench . car .' -c

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 \
        --texts 'There are a lot of cars here.'

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 \
        --texts '$: coco'

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 \
        --texts '$: lvis' --pred-score-thr 0.7 \
        --palette random --chunked-size 80

        python demo/image_demo.py demo/demo.jpg \
        grounding_dino_swin-t_pretrain_obj365_goldg_cap4m \
        --texts '$: lvis' --pred-score-thr 0.4 \
        --palette random --chunked-size 80

        python demo/image_demo.py demo/demo.jpg \
        grounding_dino_swin-t_pretrain_obj365_goldg_cap4m \
        --texts "a red car in the upper right corner" \
        --tokens-positive -1

    Visualize prediction results::

        python demo/image_demo.py demo/demo.jpg rtmdet-ins-s --show

        python demo/image_demo.py demo/demo.jpg rtmdet-ins_s_8xb32-300e_coco \
        --show
"""

import ast
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes

# RTMDet Export
import torch, cv2
import numpy as np
nn = torch.nn


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    # Once you input a format similar to $: xxx, it indicates that
    # the prompt is based on the dataset class name.
    # support $: coco, $: voc, $: cityscapes, $: lvis, $: imagenet_det.
    # detail to `mmdet/evaluation/functional/class_names.py`
    parser.add_argument(
        '--texts', help='text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    # only for GLIP and Grounding DINO
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')
    parser.add_argument(
        '--chunked-size',
        '-s',
        type=int,
        default=-1,
        help='If the number of categories is very large, '
        'you can specify this parameter to truncate multiple predictions.')
    # only for Grounding DINO
    parser.add_argument(
        '--tokens-positive',
        '-p',
        type=str,
        help='Used to specify which locations in the input text are of '
        'interest to the user. -1 indicates that no area is of interest, '
        'None indicates ignoring this parameter. '
        'The two-dimensional array represents the start and end positions.')

    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    if call_args['texts'] is not None:
        if call_args['texts'].startswith('$:'):
            dataset_name = call_args['texts'][3:].strip()
            class_names = get_classes(dataset_name)
            call_args['texts'] = [tuple(class_names)]

    if call_args['tokens_positive'] is not None:
        call_args['tokens_positive'] = ast.literal_eval(
            call_args['tokens_positive'])

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args

class onnx_tracer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output[0][0], output[0][1], output[0][2],output[1][0],output[1][1],output[1][2]

def parse(cls_input, dis_input):
    labels = torch.argmax(cls_input, dim=0)
    vals, _ = torch.max(cls_input, dim=0)
    thresh_mask = torch.sigmoid(vals) >= 0.5

    trans = dis_input.permute(1, 2, 0)

    print(labels[thresh_mask])
    print(trans[thresh_mask])


def main():
    init_args, call_args = parse_args()
    # TODO: Video and Webcam are currently not supported and
    #  may consume too much memory if your input folder has a lot of images.
    #  We will be optimized later.
    inferencer = DetInferencer(**init_args)

    # Load and pad image
    img = np.array(cv2.imread("demo/demo.jpg", cv2.IMREAD_COLOR))
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    scale = 640 / h if h > w else 640 / w
    h = int(h * scale)
    w = int(w * scale)
    pad = (w // 32 + 1) * 32 - w if h > w else (h // 32 + 1) * 32 - h

    img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
    padded_img = np.zeros((h, pad + w, c), dtype=np.uint8) if h > w else np.zeros((pad + h, w, c), dtype=np.uint8)

    padded_img[:h,:w,::] += img
    cv2.imshow("image", padded_img)
    cv2.waitKey(0)

    # preprocessing
    input_float = padded_img.astype(np.float32)
    mean = np.array([103.53, 116.28, 123.675])
    std =  np.array([57.375, 57.12,  58.395 ])
    input_float -= mean
    input_float /= std

    # Inference
    rtmdet = onnx_tracer(inferencer.model)

    input_img = torch.Tensor(input_float)
    input_img = input_img.permute(2, 0, 1).unsqueeze(0)
    cls8, cls16, cls32, dis8, dis16, dis32 = rtmdet(input_img)

    cls8 = cls8.squeeze(0)
    cls16 = cls16.squeeze(0)
    cls32 = cls32.squeeze(0)

    dis8 = dis8.squeeze(0)
    dis16 = dis16.squeeze(0)
    dis32 = dis32.squeeze(0)

    print()
    print("Model cls output")
    print(cls8.shape)
    print(cls16.shape)
    print(cls32.shape)

    print()
    print("Model box output")
    print(dis8.shape)
    print(dis16.shape)
    print(dis32.shape)

    print()
    print("Result cls8")
    parse(cls8, dis8)
    print("Result cls16")
    parse(cls16, dis16)
    print("Result cls32")
    parse(cls32, dis32)

    torch.onnx.export(
        rtmdet,                  # model to export
        input_img,        # inputs of the model,
        "rtm_export.onnx",        # filename of the ONNX model
        input_names=["input"],  # Rename inputs for the ONNX model
    )
    print("ONNX export")

if __name__ == '__main__':
    main()