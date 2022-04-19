# conda env list
# conda activate nieva_TFM

import argparse
import os
import cv2
import math
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from model import ModelSpatial
from utils import imutils, evaluation
from config import *

labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
          'hair drier', 'toothbrush']  # class names

"""
parser = argparse.ArgumentParser()
parser.add_argument('--model_weights', type=str, help='model weights', default='./models/gaze_follow/model_demo.pt')
# parser.add_argument('--image_dir', type=str, help='images', default='data/gif_cocina_avocado/frames')
parser.add_argument('--image_dir', type=str, help='images',
                    default='/home/alvaro.nieva/Documents/yolov5/runs/detect/gif_avocado_yolo')
parser.add_argument('--head', type=str, help='head bounding boxes', default='data/gif_cocina_avocado/person1.txt')
parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='arrow')
parser.add_argument('--out_threshold', type=int, help='out-of-frame target decision threshold', default=100)

parser.add_argument('--iou_punto', type=int, help='mira a un objeto detectado', default=1)
parser.add_argument('--label_dir', type=str, help='posicion de las labels', default='/home/alvaro.nieva/Documents'
                                                                                    '/yolov5/runs/detect'
                                                                                    '/gif_avocado_yolo/labels/')
args = parser.parse_args()"""
parser = argparse.ArgumentParser()
parser.add_argument('--model_weights', type=str, help='model weights', default='./models/gaze_follow/model_demo.pt')
parser.add_argument('--image_dir', type=str, help='images', default='/home/alvaro.nieva/Documents/yolov5/runs/detect/gif_avocado_yolo')
parser.add_argument('--head', type=str, help='head bounding boxes', default='data/gif_cocina_avocado/person1.txt')
parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='arrow')
parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=100)

parser.add_argument('--save_fig', type=int, help='guarda las imag', default=0)
parser.add_argument('--results_dir', type=str, help='posicion de las labels', default='resultados/primer_run_avocado_hm/')
parser.add_argument('--iou_punto', type=int, help='mira a un objeto detectado', default=1)
parser.add_argument('--label_dir', type=str, help='posicion de las labels', default='/home/alvaro.nieva/Documents'
                                                                                    '/yolov5/runs/detect'
                                                                                    '/gif_avocado_yolo/labels/')
args = parser.parse_args()


def _get_transform():
    # Hace un resize y normaliza la imagen ¿Por qué se normaliza?
    # Normalization helps get data within a range and reduces the skewness which helps learn faster and better.
    # Normalization can also tackle the diminishing and exploding gradients problems.

    transform_list = []

    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    # En el resize, no acabo de ver de donde sacar las variables input_resolution

    transform_list.append(transforms.ToTensor())
    # ToTensor converts a PIL image or NumPy ndarray into a FloatTensor.
    # and scales the image’s pixel intensity values in the range [0, 1.]
    # Pasas a tensor porque normalize solo se puede utilizar en tipo Tensor

    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    # De donde puede sacar estos valores? Los ha sacado de un tutorial de Pytorch

    return transforms.Compose(transform_list)
    # Permite encadenar todas las transformaciones anteriores


def run():
    column_names = ['frame', 'left', 'top', 'right', 'bottom']
    df = pd.read_csv(args.head, names=column_names, index_col=0)
    df['left'] -= (df['right'] - df['left']) * 0.1
    df['right'] += (df['right'] - df['left']) * 0.1
    df['top'] -= (df['bottom'] - df['top']) * 0.1
    df['bottom'] += (df['bottom'] - df['top']) * 0.1

    # set up data transformation
    test_transforms = _get_transform()

    model = ModelSpatial()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()
    model.train(False)

    # plt.ion()
    fig = plt.figure()

    with torch.no_grad():
        for i in df.index:
            frame_raw = Image.open(os.path.join(args.image_dir, i))
            frame_raw = frame_raw.convert('RGB')
            width, height = frame_raw.size

            head_box = [df.loc[i, 'left'], df.loc[i, 'top'], df.loc[i, 'right'], df.loc[i, 'bottom']]

            head = frame_raw.crop((head_box))  # head crop

            head = test_transforms(head)  # transform inputs
            frame = test_transforms(frame_raw)
            head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width,
                                                        height,
                                                        resolution=input_resolution).unsqueeze(0)
            # Da igual pasarle la bounding-box de la cara en porcentuales o totales, en la fun de arriba lo arregla.

            head = head.unsqueeze(0).cuda()
            # .Cuda pasa de cpu Tensor to gpu tensor
            # Unsqueeze cambia las dimensiones del "array" (en este caso un tensor gpu), como pone 0 se queda en 1D

            frame = frame.unsqueeze(0).cuda()

            head_channel = head_channel.unsqueeze(0).cuda()

            # forward pass
            raw_hm, _, inout = model(frame, head_channel, head)

            # heatmap modulation
            raw_hm = raw_hm.cpu().detach().numpy() * 255
            # To go from a Tensor that requires_grad to one that does not, use .detach()
            # To go from a gpu Tensor to cpu Tensor, use .cpu().
            # Tp gp from a cpu Tensor to np.array, use .numpy().

            raw_hm = raw_hm.squeeze()
            # To squeeze a tensor, we use the torch.squeeze() method. It returns a new tensor with all the dimensions of
            # the input tensor but removes size 1. For example, if the shape of the input tensor is (M ☓ 1 ☓ N ☓ 1 ☓ P),
            # then the squeezed tensor will have the shape (M ☓ M ☓ P).

            # plt.imshow(raw_hm)
            # plt.show()

            inout = inout.cpu().detach().numpy()
            # scalar alpha which quantifies whether the person’s focus of attention is located inside or outside the
            # frame. The modulation is performed by an element-wise subtraction of the (1−alpha) from the normalized
            # full-sized feature map, followed by clipping of the heatmap such that its minimum values are ≥ 0
            # Por lo tanto, inout es un solo número de 0 a 1

            inout = 1 / (1 + np.exp(-inout))
            inout = (1 - inout) * 255

            size_cv = (width, height)
            norm_map = cv2.resize(raw_hm, dsize=size_cv) - inout
            # plt.imshow(norm_map)
            # plt.show()

            # vis
            plt.clf()
            # clear figure: asi no va dando saltos la imagen

            # fig = plt.figure()
            fig.canvas.manager.window.move(0, 0)
            plt.axis('off')
            plt.imshow(frame_raw)

            ax = plt.gca()
            rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2] - head_box[0], head_box[3] - head_box[1],
                                     linewidth=2, edgecolor=(0, 1, 0), facecolor='none')
            ax.add_patch(rect)

            # Calculo de zona observada
            if inout < args.out_threshold:  # in-frame gaze, va de 0 a 255, con un treshold de 100
                pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                # saca el valor maximo del del mapa de calor (64,64)

                norm_p = [pred_x / output_resolution, pred_y / output_resolution]
                # pasa de unidades normalizadas a numero de pixels [0.53125, 0.453125]

                coor_y = norm_p[1] * size_cv[1]
                coor_x = norm_p[0] * size_cv[0]
                # coord del punto observado, tamaño de imagen

                # read labels from YOLO .txt (txt per image)
                name_txt = args.label_dir + i.split('.')[0] + '.txt'
                column_names_yolo = ['class', 'center_x', 'center_y', 'width', 'height']
                df_yolo = pd.read_csv(name_txt, names=column_names_yolo, delimiter=" ")

                # Check si el punto observado esta dentro de un objeto detectado
                for j in df_yolo.index:
                    # RangeIndex(start=0, stop=12, step=1)
                    center_x = df_yolo['center_x'][j] * size_cv[0]
                    center_y = df_yolo['center_y'][j] * size_cv[1]
                    width_yolo = df_yolo['width'][j] * size_cv[0]
                    height_yolo = df_yolo['height'][j] * size_cv[1]
                    if (center_x + width_yolo / 2) >= coor_x >= (center_x - width_yolo / 2):
                        if (center_y + height_yolo / 2) >= coor_y >= (center_y - height_yolo / 2):
                            # print(df_yolo['class'][j])
                            zona = labels[df_yolo['class'][j]]
                            titulo = 'Está mirando a: ' + zona
                            ax.set_title(titulo, fontsize=15)

            # str(inout)
            fig.suptitle(str(inout), fontsize=14, fontweight='bold')

            # Pintar la flecha o el heat map
            if args.vis_mode == 'arrow':
                if inout < args.out_threshold:  # in-frame gaze, va de 0 a 255, con un treshold de 100
                    # bola amarilla
                    circ = patches.Circle((norm_p[0] * width, norm_p[1] * height), height / 50.0, facecolor=(1, 1, 0),
                                          edgecolor='none')
                    ax.add_patch(circ)  # Linea verde
                    plt.plot((norm_p[0] * width, (head_box[0] + head_box[2]) / 2),
                             (norm_p[1] * height, (head_box[1] + head_box[3]) / 2), '-', color=(0, 1, 0, 1))
            else:
                plt.imshow(norm_map, cmap='jet', alpha=0.3, vmin=0, vmax=255)  # solo imprime en la imagen
                # Valor original de alpha=0,2, así se ve sin filtro de color la imagen

            if args.save_fig == 1:
                file = str(args.results_dir + i)
                plt.savefig(file)

            plt.show(block=False)  # enseña la imagen
            # plt.show()
            plt.pause(0.2)

        print('DONE!')


if __name__ == "__main__":
    run()
