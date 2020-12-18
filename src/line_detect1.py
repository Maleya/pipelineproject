#!/usr/bin/env python

"""
METHOD 1:
information about how this works here.
/Martti will fill in more here
"""


import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple
from collections import deque
import buffer


def param_to_points(rho, theta):
    """
    helper function:
    returns two sets of points on the line paramtrized by (rho,theta)
    """
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    p1, p2 = (x1, y1), (x2, y2)
    if y2 > y1:  # keeps the line from flipping
        p1, p2 = p2, p1

    return (p1, p2)


def is_vertical(theta, tol_deg):
    """
    Checkes if angle theta is within plus-minus tol_deg or in other words:
    if line is vertical when parametrized (rho,theta) in orgin top left.

    args:
        theta   (float): angle in radians
        tol_deg (float): allowable tolerance in degrees
    returns:
        Boolean result
    """
    tol_rad = np.pi / 180 * tol_deg
    lower_low = 0
    lower_high = tol_rad
    higher_low = np.pi - tol_rad
    higher_high = np.pi

    if lower_low <= theta <= lower_high or higher_low <= theta <= higher_high:
        return True

    return False


def pre_process(img):
    """
    Applies preprocessing steps
    """
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # thresholding: % of maxbrightness pixels to allow through:
    cutoff = 0.6
    maxval, minval = np.max(img), np.min(img)
    img[img < int(maxval * cutoff)] = 0
    img[img >= int(maxval * cutoff)] = 255

    # dialate and erode
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=2)
    # img = cv2.erode(img, kernel, iterations=2)

    return img


def get_lines(img, n_best, tol, accu_thresh=15):
    """
    returns the subset of lines within tolerance out of the n best
    lines (most accumulator votes). If none are found: returns None.

    args:
        img             (ndarray)   image
        n_best          (int)       number of lines to return
        tol             (float)     allowed deviance from vertical in degrees
        accu_thresh     (int)       min votes to consider in accumulator
    """
    line = namedtuple("Line", "rho theta")
    approved_lines = []
    lines = cv2.HoughLines(img, 1, np.pi / 180, accu_thresh)

    if lines is not None:
        for i in range(min(len(lines), n_best)):
            rho, theta = lines[i][0]
            within_tol = is_vertical(theta, tol)

            if within_tol is True:
                approved_lines.append(line(rho, theta))

    if len(approved_lines) > 0:
        return approved_lines
    else:
        return None


def main(img_deque):
    """
    TODO: Put a helpful docstring here
    inputs:
        img_deque    (collections.deque) history prior to active frame

    outputs:
        final_points (?)
        final_image  (?)
    """

    # PROCESS PARAMETERS:
    n_best = 5  # consider top n best lines
    degree_tol = 20  # permitted divergence from vertical in degrees
    acc_thresh = 15  # min votes to consider in accumulator 15
    colour = namedtuple("Colour_rbg", "R G B")
    palette = [colour(100, 255, 0), colour(6, 154, 243)]
    palette = [colour(255, 255, 255), colour(6, 154, 243)]
    weighting = "UNIFORM"
    Extend_line = False

    if len(img_deque) > 0:
        h, w = img_deque[0].shape

    image_buffer = buffer.Buffer(bf_length=len(img_deque), w_mode=weighting, img_size=(h, w))
    image_buffer.extend_line = Extend_line

    for frame in img_deque:

        final_result = np.copy(frame)
        img_WIP = np.copy(frame)

        pre_img = pre_process(img_WIP)
        good_lines = get_lines(pre_img, n_best, tol=degree_tol, accu_thresh=acc_thresh)
        img = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2RGB)  # for colorful lines

        if good_lines is not None:
            line = good_lines[0]
            points = param_to_points(line.rho, line.theta)

        else:
            points = ((None, None), (None, None))

        image_buffer.add_points(points)

    buffered_points = image_buffer.get_points()
    p1, p2 = buffered_points

    if None in [elem for tupl in buffered_points for elem in tupl]:
        cv2.line(final_result, p1, p2, palette[0], 2)

    return final_result, buffered_points


if __name__ == "__main__":
    main()
