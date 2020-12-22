"""
Project course SD2711 Small Craft Design, KTH 2020.
Examiner: Jakob Kuttenkeuler
Supervisor: Aldo Teran Espinoza

METHOD 1:
for each frame do: 
- Thresholding max intensity
- erotion and dialation filters
- collection of best lines from Hough transoform
- filter away lines deviating from degree tolerance

Sending multiple frames to main() returns a weighted result over all frames,
either uniformly or linearly. See main for more information.
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
    Helper function:
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

    args:
        img     (np.ndarray) imagefile
    returns:
        img     (np.ndarray) processed file
    """

    # percentage of maxbrightness pixels to allow through:
    cutoff = 0.6
    maxval, minval = np.max(img), np.min(img)
    img[img < int(maxval * cutoff)] = 0
    img[img >= int(maxval * cutoff)] = 255

    # dialate and erode operations
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
    Accepts a listlike iterable of images and returns a best guess line
    weighted on all frames in img_deque.

    Can either be weighted uniformly: w = 1/batchsize
    or linearly as: w = np.linspace(0,1,num=batchsize)
    thereby placing most weight on the latest frame

    args:
        img_deque       (collections.deque) history prior to active frame
    returns:
        buffered_points (tuple) returns points in form ((x1,y1),(x2,y2))
        final_result    (np.ndarray) latest image frame with line overlay
    """

    # PROCESS PARAMETERS:
    n_best = 5  # consider top n best lines
    degree_tol = 20  # permitted divergence from vertical in degrees
    acc_thresh = 15  # min votes to consider in accumulator 15
    colour = namedtuple("Colour_rbg", "R G B")
    palette = [colour(255, 255, 255), colour(6, 154, 243)]
    weighting = "UNIFORM"  # weighting mode, see above
    Extend_line = False  # extends distance between points for plotting

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

    # draw the line
    if None not in [elem for tupl in buffered_points for elem in tupl]:
        cv2.line(final_result, p1, p2, palette[0], 2)

    return final_result, buffered_points


if __name__ == "__main__":
    main()
