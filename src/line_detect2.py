import cv2
import numpy as np
from sklearn.decomposition import PCA
from buffer import Buffer

def preprocess(image, mode="DIL"):
    """
    Preprocessing image with:
        - Thresholding
        - Erotion & Dilation: used to remove noise
            number of iterations determine how much to erode/dilate.
        - Contouring
    """
    if mode == "DIL":
        mask = threshold(image, threshold=0.7)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=3)
        threshold_img = cv2.dilate(erosion, kernel, iterations=1)
    elif mode == "DO_NOT_DILATE":
        threshold_img = threshold(image, threshold=0.8)
    else:
        raise KeyError("Mode not yet implemented.")
    idx_white = np.where(threshold_img == 255)
    mu, sigma = get_Gaussian_params(idx_white)

    return mu, sigma, idx_white


def threshold(image, threshold):
    maxval = np.max(image)
    image[image < int(maxval * threshold)] = 0
    image[image >= int(maxval * threshold)] = 255
    return image 


def get_Gaussian_params(idx_white):
    """
    Computes center and spread of the contours in two directions x and y.
    """

    if len(idx_white[0]) > 1:
        mu_y = np.mean(idx_white[0])
        mu_x = np.mean(idx_white[1])
        sigma_xy = np.cov(idx_white[1].transpose(), idx_white[0].transpose())

        mu = np.array([mu_x, mu_y])
    else:
        mu, sigma_xy = None, None
    return mu, sigma_xy


def detect_line(image_):
    """
    This line detector has two modes, (DO_NOT_DILATE, DIL). To find a line it is included to:
        - check if any white pixels were found for the modes.
        - check if the countour center is within a threshold for the modes.
        - check the ratio of spread  = (spread in vertical direction) / (spread in horizontal direction) for the modes.
        - make a decision if dilation should be used or not.
        - perform PCA to fit a line.
        - check if the fitted line is within a threshold of +-20 degrees.
        - check if the spread in the vertical direction > the spread in the horizontal direction.
    Otherwise: No line detected.

    return: 
        - processed_img, processed image
        - p_hat, points to be fitted
    """
    image = np.copy(image_)
    h, w = image.shape
    mode_mask = "DO_NOT_DILATE"
    mode_dil = "DIL"

    mu_mask, sigma_mask, idx_white_mask = preprocess(
        image, mode=mode_mask)
    mu_dil, sigma_dil, idx_white_dil = preprocess(
        image, mode=mode_dil)

    isarray_mask, ratio_mask = validity_check(
        sigma_mask, mode=mode_mask)
    isarray_dil, ratio_dil = validity_check(sigma_dil, mode=mode_dil)

    if not isarray_mask and not isarray_dil:
        points = ((None, None), (None, None))
        txt = "No line detected"
    else:
        if ratio_mask > ratio_dil or not isarray_dil:
            idx_white = idx_white_mask
            sigma = sigma_mask
            mu = mu_mask
            method = "(b)"
        elif isarray_dil:
            idx_white = idx_white_dil
            sigma = sigma_dil
            mu = mu_dil
            method = "(a)"

        points = get_PCA(image, idx_white)
        dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]
        valid_len = point_vector_length_check(dx, dy)
        isvalid_line = valid_heading_angle(dx, dy)
        if sigma[1, 1] > 1.5*sigma[0, 0] and isvalid_line and valid_len:
            txt = "Line detection\nusing PCA and " + method
        else:
            points = ((None, None), (None, None))
            txt = "No line detected"

    return points, txt


def point_vector_length_check(dx, dy):
    tol = 3
    length = np.sqrt(dx**2+dy**2)
    if length >= tol:
        return True
    else:
        return False


def validity_check(sigma, mode):
    if mode == "DIL":
        val = 1e10
    elif mode == "DO_NOT_DILATE":
        val = -1e10
    else:
        raise KeyError("Mode not yet implemented.")

    isarray = isinstance(sigma, np.ndarray)
    if isarray and sigma[0, 0] != 0:
        ratio = sigma[1, 1] / sigma[0, 0]
    else:
        ratio = val

    return isarray, ratio


def valid_heading_angle(dx, dy):
    theta = np.rad2deg(np.arctan2(dy, dx))
    tol = 25
    lb, ub = 90-tol, 90+tol
    if (np.abs(theta) >= lb and np.abs(theta) <= ub):
        return True
    else:
        return False


def get_PCA(image, idx_white):
    n_points = 2
    data = np.zeros((len(idx_white[0]), n_points))
    data[:, 0] = idx_white[1]
    data[:, 1] = idx_white[0]

    pca = PCA(n_components=1)
    pca.fit(data)
    p_hat = np.zeros((n_points, n_points))
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        p_hat[0, :] = pca.mean_
        p_hat[1, :] = pca.mean_ + v

    if p_hat[1, 1] >= p_hat[0, 1]:
        p_hat[0, :] = pca.mean_ + v
        p_hat[1, :] = pca.mean_
    return ((p_hat[0, 0], p_hat[0, 1]), (p_hat[1, 0], p_hat[1, 1]))


def main(img_queue):

    if len(img_queue) > 0:
        size = np.shape(img_queue[0])

    buff = Buffer(bf_length=len(img_queue), img_size=size, w_mode="LINEAR")
    buff.extend_line = True
    for image in img_queue:
        img_init = np.copy(image)
        points, txt = detect_line(img_init)  # (792, 304, 3)
        buff.add_points(points)

    p0, p1 = buff.get_points()
    if p0[0] is None:
        final_result = img_init
        p0, p1 = (0., 0.), (0., 0.)
    else:
        final_result = cv2.line(img_init, pt1=(p0[0], p0[1]), pt2=(p1[0], p1[1]), color=(255, 0, 0), thickness=2)

    return final_result, (p0, p1)


if __name__ == "__main__":
    main()
