# Project course SD2711 Small Craft Design, KTH 2020.
# Examiner: Jakob Kuttenkeuler
# Supervisor: Aldo Teran Espinoza
import numpy as np


class Buffer:
    def __init__(self, bf_length=30, img_size=(416, 256), w_mode="LINEAR"):
        self.buffer = np.zeros((4, bf_length))
        self.w_mode = w_mode
        self.extend_line = False
        self.count = 0
        self.h, self.w = img_size

    def add_points(self, points):
        """
        Adds the last points
        Input:
            - points (numpy.array)
        """
        self.buffer[:2, self.count] = points[0]
        self.buffer[2:, self.count] = points[1]
        self.count += 1

    def get_valid_buffer(self):
        """
        Removes nan-points from the buffer to get valid points in buffer
        Return: 
            - n (int): number of valid points
            - valid_buffer (numpy.array): buffer without nan-points
        """
        valid_col_idx = np.where(~np.isnan(self.buffer[0]))[0]
        valid_points, n = self.any_valid_points(valid_col_idx)

        if valid_points:
            valid_buffer = np.zeros((4, n))
            for i, col in enumerate(valid_col_idx):
                valid_buffer[:, i] = self.buffer[:, col]
        else:
            valid_buffer = self.buffer[:, 0]
        return n, valid_buffer

    def any_valid_points(self, idx):
        """
        check if buffer has valid points
        Input:
            - idx (numpy.array): indicies of valid points in buffer
        Return: 
            - True/False (bool)
            - n (int): number of valid points in buffer
        """
        n = len(idx)
        if n > 0:
            return True, n
        else:
            return False, 0

    def get_points(self):
        """
        computes weighted points from the buffer
            * uniform weights
            * ranked weights where the latest point from the buffer has the largest weight and the oldest has the lowest
        Return: 
            - ((p00,p01), (p10, p11)), (tuple): new points 
        """
        # get valid buffer without nan-points
        n, valid_buffer = self.get_valid_buffer()
        if n == 0:
            return ((None, None), (None, None))
        else:
            # uniform weighting
            if self.w_mode == "UNIFORM":
                p00, p01, p10, p11 = np.mean(valid_buffer, axis=1)
            # ranked weights
            elif self.w_mode == "LINEAR":
                W = np.linspace(1, n, n)
                norm = np.sum(W)
                # normalising weights such that sum(w)=1
                W = W/norm
                p00, p01, p10, p11 = np.dot(valid_buffer, W)
            else:
                raise KeyError("Mode not yet Implemented.")

            # extend line if the line is short
            if self.extend_line:
                # Compute the lenght of the line
                dy = p11 - p01
                dx = p10 - p00

                # Decide the line extension dependent on the line length
                if dy < self.h/10:
                    ky = 100
                elif dy >= self.h / 10 and dy < self.h / 5:
                    ky = 50
                else:
                    ky = 20
                if dx < self.w/10:
                    kx = 100
                elif dx >= self.w / 10 and dx < self.w / 5:
                    kx = 50
                else:
                    kx = 20
                # Computes new cordinates in gradient direction
                # (x0,y0)
                p00 -= kx * dx
                p01 -= ky * dy
                # (x1,y1)
                p10 += kx * dx
                p11 += ky * dy

            return ((int(p00), int(p01)), ((int)(p10), int(p11)))


if __name__ == "__main__":
    """
    Use the buffer class like below
    """
    bf = Buffer(bf_length=5)

    p = (1, 2, 3, 4)
    p_w_nones = (None, None, None, None)
    for i in range(5):
        if i == 3 or i == 2:
            bf.add_points(p_w_nones)
        else:
            bf.add_points(p)
    points = bf.get_points()
