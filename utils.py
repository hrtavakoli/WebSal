'''
utility functions for handling images

@author: Hamed R. Tavakoli
'''

import cv2
import numpy as np
from PIL import Image


def padded_resize(image, shape_r, shape_c):

    new_image = Image.new("RGB", (shape_c, shape_r))
    shape = image.size

    ratio_c = shape[0] / shape_c
    ratio_r = shape[1] / shape_r

    new_cols = shape_c
    new_rows = shape_r
    if ratio_r > ratio_c:
        new_cols = (shape[0] * shape_r) // shape[1]

    if ratio_r < ratio_c:
        new_rows = (shape[1] * shape_c) // shape[0]

    image = image.resize((new_cols, new_rows), Image.ANTIALIAS)

    new_image.paste(image, ((shape_c - new_cols) // 2,
                            (shape_r - new_rows) // 2))
    return new_image


def resize_fixation_map(img, rows, cols):

    shape = img.size

    new_tmp = np.zeros((rows, cols))

    ratio_r = rows / shape[1]
    ratio_c = cols / shape[0]

    tmp = np.asarray(img)
    coords = np.argwhere(tmp)
    for coord in coords:
        r = int(np.round(coord[0]*ratio_r))
        c = int(np.round(coord[1]*ratio_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        new_tmp[r, c] = 255

    return Image.fromarray(new_tmp)


def resize_padded_fixation(image, shape_r, shape_c):

    new_image = Image.new("L", (shape_c, shape_r))
    shape = image.size

    ratio_c = shape[0] / shape_c
    ratio_r = shape[1] / shape_r

    new_cols = shape_c
    new_rows = shape_r
    if ratio_r > ratio_c:
        new_cols = (shape[0] * shape_r) // shape[1]

    if ratio_r < ratio_c:
        new_rows = (shape[1] * shape_c) // shape[0]

    image = resize_fixation_map(image, new_rows, new_cols)

    new_image.paste(image, ((shape_c - new_cols) // 2,
                            (shape_r - new_rows) // 2))
    return new_image


def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    pred = pred / np.max(pred) * 255

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    img = img / np.max(img) * 255

    return img