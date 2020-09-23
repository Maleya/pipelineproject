import math
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
ideas:  
- Run hough with sobel-gradients (paper)
- Run hough with canny edge detection (logical) [DONE]
- Guess angle by averaging [DONE


todo: 
- make a pipeline of all the images in /img/  [DONE]
- try multiscale hough trnsform 
- try the probabilistic hough(different output)

"""


def load_images(folder):
    """
    loads images from folder
    """
    image_list = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            image_list.append(img)
    print(f"{len(image_list)} images loaded")
    return image_list


def RBG_to_intensity(image, weights=(0.30, 0.59, 0.11)):
    """
    Converts RGB image to greyscale according to weights=(r,g,b)
    """
    x, y, _ = image.shape
    intensity = np.zeros((x, y))
    for i in range(3):
        scaled = image[:, :, i] * weights[i]
        intensity = intensity + scaled

    return intensity


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

    return ((x1, y1), (x2, y2))


img_list = load_images("img")
img_len = len(img_list)
output = []

for j, img in enumerate(img_list):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5),10)
    canny = cv2.Canny(img, 30, 80, 3, L2gradient=True)
    lines = cv2.HoughLines(canny, 1, np.pi / 180, 100)

    if lines is not None:
        for i in range(len(lines)):
            rho, theta = lines[i][0]
            p1, p2 = param_to_points(rho, theta)
            # print(f"rho={rho}, theta={theta}")
            # print(f"rho={rho}, theta={theta*180/np.pi}")
            cv2.line(img, p1, p2, (255, 0, 0), 2)

    # plot average line:
    avg_rho = np.median(lines[:, 0, 0])
    avg_theta = np.median(lines[:, 0, 1])
    p1, p2 = param_to_points(avg_rho, avg_theta)
    cv2.line(img, p1, p2, (0, 0, 255), 2)

    output.append(img)
    output.append(canny)


# plots:
nrows, ncols = 2, 3  # array of sub-plots
fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

for i, axi in enumerate(ax.flat):
    axi.imshow(output[i], cmap="gray")
    # get indices of row/column
    rowid = i // ncols
    colid = i % ncols

    # axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))
    axi.set_xticks([]), axi.set_yticks([])

# # one can access the axes by ax[row_id][col_id]
# # do additional plotting on ax[row_id][col_id] of your choice
plt.tight_layout()
plt.show()
