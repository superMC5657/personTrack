# -*- coding: utf-8 -*-
# !@time: 2020/6/10 上午10:59
# !@author: superMC @email: 18758266469@163.com
# !@fileName: image_tool.py

import cv2
import numpy as np
from self_utils.chinese_text import Chinese_text
from PIL import Image, ImageDraw, ImageFont
from config import opt
from self_utils.utils import get_color

ft = Chinese_text('data/font/HuaWenHeiTi-1.ttf')
colors = get_color()
font = ImageFont.truetype("data/font/HuaWenHeiTi-1.ttf", 15, encoding="utf-8")


def plot_boxes(image, persons):
    im_height, im_width, _ = image.shape
    scale = 12
    persons = sorted(persons, key=lambda x: x.id)
    for i in range(len(persons)):
        pBox = persons[i].pBox
        color = colors[persons[i].id]
        image = cv2.rectangle(image, (pBox[0], pBox[1]), (pBox[2], pBox[3]), color, 2)
        fBox = persons[i].fBox
        if fBox is not None:
            image = cv2.rectangle(image, (fBox[0], fBox[1]), (fBox[2], fBox[3]), color, 2)
        image = ft.draw_text(image, str(persons[i].id) + " " + persons[i].name, (pBox[0] + 5, pBox[3] - 5), scale,
                             color)

        image = ft.draw_text(image,
                             str(persons[i].id) + " " + persons[i].name + " " + str(int(persons[i].time)),
                             (im_width - opt.wight_padding + 5, im_height - 20 * persons[i].id), scale,
                             color)
    return image


def plot_boxes_pil(image, persons):
    im_height, im_width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    line_width = 2

    persons = sorted(persons, key=lambda x: x.id)
    for i in range(len(persons)):
        pBox = persons[i].pBox
        # color = "white"
        color = colors[persons[i].id]
        draw.rectangle((pBox[0], pBox[1], pBox[2], pBox[3]), outline=color, width=line_width)
        fBox = persons[i].fBox
        if fBox is not None:
            draw.rectangle((fBox[0], fBox[1], fBox[2], fBox[3]), outline=color, width=line_width)
        draw.text((pBox[0] + 5, pBox[3] + 5),
                  str(persons[i].id) + " " + str(persons[i].name) + " " + str(round(persons[i].fid_min_distance, 2)),
                  fill=color, font=font)
        draw.text((im_width - opt.wight_padding + 5, im_height - 20 * persons[i].id),
                  str(persons[i].id) + " " + str(persons[i].name) + " " + str(int(persons[i].time)),
                  fill=color, font=font)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image
