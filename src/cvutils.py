import numpy as np
import cv2

# calibration result (Pickle) file name
calfilename = "../pickle/cal_result.p"

# binary filtered image conbination parameters file name
paramfilename = "../pickle/combination_parameters.p"

def undistort_image(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary


def hls_s_select(hls, thresh=(0, 255)):
    S = hls[:, :, 2]
    sxbinary = np.zeros_like(S)
    sxbinary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return sxbinary

def hls_l_select(hls, thresh=(0, 255)):
    L = hls[:, :, 1]
    sxbinary = np.zeros_like(L)
    sxbinary[(L > thresh[0]) & (L <= thresh[1])] = 1
    return sxbinary

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel )
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = cv2.convertScaleAbs( sobelx )
    abs_sobely = cv2.convertScaleAbs( sobely )
    graddir = np.arctan2(abs_sobely, abs_sobelx)
    sxbinary = np.zeros_like(graddir)
    sxbinary[(graddir >= thresh[0]) & (graddir <= thresh[1])] = 1
    return sxbinary

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return sxbinary

def gray_thresh(gray, thresh=(0, 255)):
    sxbinary = np.zeros_like(gray)
    sxbinary[(gray >= thresh[0]) & (gray <= thresh[1])] = 1
    return sxbinary
