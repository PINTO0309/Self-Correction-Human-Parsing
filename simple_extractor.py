#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import sys
import copy
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='', help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")
    parser.add_argument("--onnx-export-only", action='store_true')

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

class SegmentModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, input_size=[512, 512]):
        super(SegmentModel, self).__init__()
        self.model = model
        self.input_size = input_size

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        output = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)(outputs[0][-1])
        output = torch.argmax(output, dim=1, keepdim=True)
        return output

def main():
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    if not args.onnx_export_only:
        model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
        state_dict = torch.load(args.model_restore)['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.cuda()

    if args.onnx_export_only:
        import onnx
        from onnxsim import simplify
        RESOLUTION = [
            # [192,320],
            # [192,416],
            # [192,640],
            # [192,800],
            # [256,320],
            # [256,416],
            # [256,448],
            # [256,640],
            # [256,800],
            # [256,960],
            # [288,480],
            # [288,640],
            # [288,800],
            # [288,960],
            # [288,1280],
            # [320,320],
            # [384,480],
            # [384,640],
            # [384,800],
            # [384,960],
            # [384,1280],
            # [416,416],

            [480,480],
            [480,640],
            [480,800],
            [480,960],
            [480,1280],
            [512,512],
            [512,640],
            [512,896],
            [544,800],
            [544,960],
            [544,1280],
            [640,640],
            [736,1280],
        ]
        for H, W in RESOLUTION:
            model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
            state_dict = torch.load(args.model_restore)['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model = SegmentModel(
                model=model,
                input_size=[H, W],
            )
            model.cpu()
            model.eval()

            onnx_file = f"bodyparse_{args.dataset}_{num_classes}_1x3x{H}x{W}.onnx"
            x = torch.randn(1, 3, H, W).cpu()
            torch.onnx.export(
                model,
                args=(x),
                f=onnx_file,
                opset_version=11,
                input_names=['input_bgr'],
                output_names=['segment'],
            )
            model_onnx1 = onnx.load(onnx_file)
            model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
            onnx.save(model_onnx1, onnx_file)
            model_onnx2 = onnx.load(onnx_file)
            model_simp, check = simplify(model_onnx2)
            onnx.save(model_simp, onnx_file)
            model_onnx2 = onnx.load(onnx_file)
            model_simp, check = simplify(model_onnx2)
            onnx.save(model_simp, onnx_file)
            model_onnx2 = onnx.load(onnx_file)
            model_simp, check = simplify(model_onnx2)
            onnx.save(model_simp, onnx_file)
        sys.exit(0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(root=args.input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    palette = get_palette(num_classes)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            parsing_result_path = os.path.join(args.output_dir, img_name[:-4] + '.png')
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(palette)
            output_img.save(parsing_result_path)
            if args.logits:
                logits_result_path = os.path.join(args.output_dir, img_name[:-4] + '.npy')
                np.save(logits_result_path, logits_result)
    return


if __name__ == '__main__':
    main()
