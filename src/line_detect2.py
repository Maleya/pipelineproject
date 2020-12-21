# Project course SD2711 Small Craft Design, KTH 2020.
# Examiner: Jakob Kuttenkeuler
# Supervisor: Aldo Teran Espinoza


import cv2
import numpy as np
from sklearn.decomposition import PCA
from buffer import Buffer


def _preprocess(image, mode="THRESH_ERODE_DILATE_IMG"):
    """
    Preprocessing image and returns a mean and standard deviation of point cloud.
    Input: 
        - image
        - mode, (str): THRESH_IMG or THRESH_ERODE_DILATE_IMG
    Return: 
        - mu (numpy.array)
        - sigma (numpy.array)
    """
    if mode == "THRESH_ERODE_DILATE_IMG":
        mask = __threshold(image, threshold=0.7)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=3)
        threshold_img = cv2.dilate(erosion, kernel, iterations=1)
    elif mode == "THRESH_IMG":
        threshold_img = __threshold(image, threshold=0.8)
    else:
        raise KeyError("Mode not yet implemented.")
    # Find white pixels in image to get pixel indicies
    idx_white = np.where(threshold_img == 255)
    mu, sigma = __get_Gaussian_params(idx_white)

    return mu, sigma, idx_white


def __threshold(image, threshold):
    """
    Threshold image
    Input: 
        - image
        - threshold (float): percentage [0,1]
    Return:
        - thresholded image
    """
    maxval = np.max(image)
    image[image < int(maxval * threshold)] = 0
    image[image >= int(maxval * threshold)] = 255
    return image


def __get_Gaussian_params(idx):
    """
    Computes center and spread of the pixel indicies in x and y direction.
    Input: 
        - idx (int): index of white pixels
    Return: 
        - mu (numpy.array): mean pixel value of point cloud
        - sigma (numpy.array): covariance of pixel values of point cloud
    """

    if len(idx[0]) > 1:
        mu_y = np.mean(idx[0])
        mu_x = np.mean(idx[1])
        sigma = np.cov(idx[1].transpose(), idx[0].transpose())

        mu = np.array([mu_x, mu_y])
    else:
        mu, sigma = None, None
    return mu, sigma


def detect_line(image_):
    """
    This line detector has two modes, (THRESH_ERODE_DILATE_IMG, THRESH_IMG). To find a line it is included to:
        * check if any white pixels were found for the modes.
        * check the ratio of spread  = (spread in vertical direction) / (spread in horizontal direction) for the modes.
        * make a decision if dilation should be used or not.
        * perform PCA to fit a line.
        * check if the fitted line is within a threshold of +-20 degrees.
        * check if the spread in the vertical direction > the spread in the horizontal direction.
    Otherwise: No line detected.
    Input:
        - image
    return: 
        - points_ (tuple): points to draw line
    """
    image = np.copy(image_)
    h, w = image.shape

    # Preprocess image by two methods
    mu_mask, sigma_mask, idx_white_mask = _preprocess(
        image, mode="THRESH_IMG")
    mu_dil, sigma_dil, idx_white_dil = _preprocess(
        image, mode="THRESH_ERODE_DILATE_IMG")

    # Check if the point cloud excist
    isarray_mask, ratio_mask = _validity_check(
        sigma_mask, mode="THRESH_IMG")
    isarray_dil, ratio_dil = _validity_check(
        sigma_dil, mode="THRESH_ERODE_DILATE_IMG")

    # If point cloud excist, do PCA and validity check of points
    if not isarray_mask and not isarray_dil:
        points_ = ((None, None), (None, None))
    else:
        if ratio_mask > ratio_dil or not isarray_dil:
            idx_white = idx_white_mask
            sigma = sigma_mask
            mu = mu_mask
        elif isarray_dil:
            idx_white = idx_white_dil
            sigma = sigma_dil
            mu = mu_dil

        # perform PCA to get two points (p0, p1)
        points = _get_PCA(image, idx_white)
        # compute dx, dy
        dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]

        # check euclidean distance of points (p0, p1)
        valid_len = _point_vector_length_check(dx, dy)
        # check angle from vertical line of points (p0, p1)
        isvalid_line = _valid_heading_angle(dx, dy)
        # if the spread is larger in vertical direction and the line has a valid bearing and length -> use PCA points
        if sigma[1, 1] > 1.5*sigma[0, 0] and isvalid_line and valid_len:
            points_ = points

        else:
            points_ = ((None, None), (None, None))

    return points_


def _point_vector_length_check(dx, dy):
    """
    Determine weather the euclidean distance of dx, dy exceed a threshold.
    Input: 
        - dx: difference of points in horizontal direction
        - dy: difference of points in vertical direction
    Return: 
        - True/Fale: (bool)
    """
    # tolerance of pixel length set to 3
    tol = 3
    length = np.sqrt(dx**2+dy**2)
    if length >= tol:
        return True
    else:
        return False


def _validity_check(sigma, mode):
    """
    Check validity that sigma_x is not equal to zero and computes the ratio of (spread in vertical direction)/ (spread in horizontal direction).
    Input: 
        - sigma, (numpy.array)
        - mode, (str)
    Return: 
        - isarray: (bool)
        - ratio: (float)
    """
    # Used to output a number of ratio without affecting performance
    if mode == "THRESH_ERODE_DILATE_IMG":
        val = 1e10
    elif mode == "THRESH_IMG":
        val = -1e10
    else:
        raise KeyError("Mode not yet implemented.")

    isarray = isinstance(sigma, np.ndarray)
    if isarray and sigma[0, 0] != 0:
        ratio = sigma[1, 1] / sigma[0, 0]
    else:
        ratio = val

    return isarray, ratio


def _valid_heading_angle(dx, dy):
    """
    Determine weather the angle between points exceed a threshold of +-25 degrees.
    Input: 
        - dx: difference of points in horizontal direction
        - dy: difference of points in vertical direction
    Return: 
        - True/Fale: (bool)
    """
    theta = np.rad2deg(np.arctan2(dy, dx))
    # threshold of +- 25 degrees
    tol = 25
    lb, ub = 90-tol, 90+tol
    if (np.abs(theta) >= lb and np.abs(theta) <= ub):
        return True
    else:
        return False


def _get_PCA(image, idx):
    """
    Determine weather the euclidean distance of dx, dy exceed a threshold.
    Input: 
        - image
        - idx (numpy.array): indicies of white pixels
    Return: 
        - ((p00, p01),(p10, p11)) (tuple): points
    """
    n_points = 2
    # create point cloud of white pixels
    data = np.zeros((len(idx[0]), n_points))
    data[:, 0] = idx[1]
    data[:, 1] = idx[0]

    # initialize PCA
    pca = PCA(n_components=1)
    pca.fit(data)

    # get PCA components
    points = np.zeros((n_points, n_points))
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        points[0, :] = pca.mean_
        points[1, :] = pca.mean_ + v

    # checks the value of vertical elements of points are in correct direction, if not flip the points -> (p0, p1) = (p1, p0)
    if points[1, 1] >= points[0, 1]:
        points[0, :] = pca.mean_ + v
        points[1, :] = pca.mean_
    return ((points[0, 0], points[0, 1]), (points[1, 0], points[1, 1]))


def main(img_queue):
    """
    Functionality: 
        1. uses n images to detect points
        2. adding points to buffer
        3. computes weighted/mean points from buffer
    Input:
        - img_queue: queue of n images 
    Return: 
        - image: image with/without points to be published
        - (p0, p1) (tuple): points to be published
    """

    if len(img_queue) > 0:
        size = np.shape(img_queue[0])

    # initialize buffer
    buff = Buffer(bf_length=len(img_queue), img_size=size, w_mode="LINEAR")
    # Choose if using line extension (has shown improved results)
    buff.extend_line = True

    # iterated through image queue to detect points and adding them to buffer
    for image in img_queue:
        img_init = np.copy(image)
        points = detect_line(img_init)  # (792, 304, 3)
        buff.add_points(points)

    # get points from buffer
    p0, p1 = buff.get_points()
    # Checks if line got detected
    if p0[0] is None:
        final_result = img_init
        p0, p1 = (0., 0.), (0., 0.)
    else:
        # draw line on image
        final_result = cv2.line(img_init, pt1=(p0[0], p0[1]), pt2=(
            p1[0], p1[1]), color=(255, 0, 0), thickness=2)

    return final_result, (p0, p1)


if __name__ == "__main__":
    main()
