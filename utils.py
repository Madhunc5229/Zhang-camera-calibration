import cv2
import numpy as np


def getWorldPoints(n_x=9, n_y=6, width=21.5):
    
    x, y = np.meshgrid(range(n_x), range(n_y))
    x, y = x.reshape(int(n_x*n_y), 1), y.reshape(int(n_x*n_y), 1)
    M = np.hstack((x, y)).astype(np.float32)
    M = M  * width

    return M


def getCorrespondces(img_path):
    M = getWorldPoints()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img, (9,6),None)
    corners = corners.reshape(-1,2)
    m = cv2.cornerSubPix(img,corners, (11,11), (-1,-1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    H,_ = cv2.findHomography(M,m)

    return M, m, H
    