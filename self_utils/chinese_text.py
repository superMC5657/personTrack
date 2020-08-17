# -*- coding: utf-8 -*-
# !@time: 2020/8/17 下午1:48
# !@author: superMC @email: 18758266469@163.com
# !@fileName: chinese_text.py
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
##################################################
# tools                                          #
#------------------------------------------------#
# draw chinese text using freetype on python2.x  #                  #
# 2017.4.12                                      #
##################################################
'''

import copy

import freetype
import numpy as np


class Chinese_text(object):
    def __init__(self, ttf):
        self._face = freetype.Face(ttf)

    def draw_text(self, image, text, pos, text_size, text_color):
        '''
        draw chinese(or not) text with ttf
        :param image:     image(numpy.ndarray) to draw text
        :param pos:       where to draw text
        :param text:      the context, for chinese should be unicode type
        :param text_size: text size
        :param text_color:text color
        :return:          image
        '''
        self._face.set_char_size(text_size * 64)
        if not isinstance(text, freetype.unicode):
            text = text.decode('utf-8')
        img = self.draw_string(image, pos[0], pos[1], text, text_color)
        return img

    def draw_string(self, img, x_pos, y_pos, text, color):
        '''
        draw string
        :param x_pos: text x-postion on img
        :param y_pos: text y-postion on img
        :param text:  text (unicode)
        :param color: text color
        :return:      image
        '''
        prev_char = 0
        pen = freetype.Vector()
        pen.x = x_pos << 6  # div 64
        pen.y = y_pos << 6

        hscale = 1.0
        matrix = freetype.Matrix(int(hscale) * 0x10000, int(0.2 * 0x10000), int(0.0 * 0x10000), int(1.1 * 0x10000))
        cur_pen = freetype.Vector()
        pen_translate = freetype.Vector()

        image = copy.deepcopy(img)
        for cur_char in text:
            self._face.set_transform(matrix, pen_translate)

            self._face.load_char(cur_char)
            kerning = self._face.get_kerning(prev_char, cur_char)
            pen.x += kerning.x
            slot = self._face.glyph
            bitmap = slot.bitmap

            cur_pen.x = pen.x
            cur_pen.y = pen.y - slot.bitmap_top * 64
            self.draw_ft_bitmap(image, bitmap, cur_pen, color)

            pen.x += slot.advance.x
            prev_char = cur_char

        return image

    def draw_ft_bitmap(self, img, bitmap, pen, color):
        '''
        draw each char
        :param bitmap: bitmap
        :param pen:    pen
        :param color:  pen color e.g.(0,0,255) - red
        :return:       image
        '''
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        cols = bitmap.width
        rows = bitmap.rows

        glyph_pixels = bitmap.buffer

        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row * cols + col] != 0:
                    img[y_pos + row][x_pos + col][0] = color[0]
                    img[y_pos + row][x_pos + col][1] = color[1]
                    img[y_pos + row][x_pos + col][2] = color[2]


if __name__ == '__main__':
    # just for test
    import cv2

    # file = open('class.txt','r')
    # line = file.readline()
    img = np.zeros([300, 300, 3])

    color_ = (0, 255, 0)
    pos = (3, 15)
    text_size = 100

    ft = Chinese_text('data/font/HuaWenHeiTi-1.ttf')
    img = ft.draw_text(img, pos, "你好的", text_size, color_)

    cv2.imshow('ss', img)
    cv2.waitKey(0)
