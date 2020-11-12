import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import cv2

# calibration result (Pickle) file name
calfilename = "../pickle/cal_result.p"

# binary filtered image conbination parameters file name
paramfilename = "../pickle/combination_parameters.p"

def gen_all_binary_images( grayimg, ghls, param ):
    blurred = cv2.GaussianBlur(grayimg, (param['kernel_size'], param['kernel_size']), 0)
    gradx_binary = abs_sobel_thresh(blurred, orient='x', thresh_min=param['gradx_min'], thresh_max=param['gradx_max'])
    grady_binary = abs_sobel_thresh(blurred, orient='y', thresh_min=param['grady_min'], thresh_max=param['grady_max'])
    mag_binary = mag_thresh(blurred, sobel_kernel=param['mag_kernel_size'], mag_thresh=(param['mag_min'], param['mag_max']))
    gray_binary = gray_thresh(blurred, thresh=(param['gray_min'], param['gray_max']))
    s_binary = hls_s_select(ghls, thresh=(param['s_min'], param['s_max']))
    l_binary = hls_l_select(ghls, thresh=(param['l_min'], param['l_max']))
    return (blurred, gray_binary, gradx_binary, grady_binary, mag_binary, s_binary, l_binary )

def combine_images(gray_binary, gradx_binary, grady_binary, mag_binary, s_binary, l_binary):
    combined = np.zeros_like(gray_binary)
    combined[((gray_binary == 1) | (gradx_binary == 1) | (mag_binary == 1) | (s_binary == 1) | (l_binary == 1)) & (grady_binary == 0)] = 1
    return combined

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

# Compute and apply perpective transform

def warper(img, warp_in_fname = '', warp_out_fname = ''):
    img_size = ( img.shape[1], img.shape[0] )

    # Source (image trapezoid) and Destination (rectangle) area coordinates. These are predefined, defined only by the camera position/angles/field of view.
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    if warp_in_fname != '':
        warp_in = img
        # it's BGR ! Red is 0,0,255 now.
        cv2.line(warp_in, ( int(src[0][0]), int(src[0][1] ) ), ( int(src[1][0]), int(src[1][1] ) ), [0, 0, 255], 3)
        cv2.line(warp_in, ( int(src[1][0]), int(src[1][1] ) ), ( int(src[2][0]), int(src[2][1] ) ), [0, 0, 255], 3)
        cv2.line(warp_in, ( int(src[2][0]), int(src[2][1] ) ), ( int(src[3][0]), int(src[3][1] ) ), [0, 0, 255], 3)
        cv2.line(warp_in, ( int(src[3][0]), int(src[3][1] ) ), ( int(src[0][0]), int(src[0][1] ) ), [0, 0, 255], 3)
        if warp_in.shape[2]!=1:
            cv2.imwrite( warp_in_fname, warp_in )
        else:
            cv2.imwrite( warp_in_fname, warp_in, cmap=cm.gray )

    if warp_out_fname != '':
        warp_out = warped
        cv2.line(warp_out, ( int(dst[0][0]), int(dst[0][1] ) ), ( int(dst[1][0]), int(dst[1][1] ) ), [0, 0, 255], 3)
        cv2.line(warp_out, ( int(dst[1][0]), int(dst[1][1] ) ), ( int(dst[2][0]), int(dst[2][1] ) ), [0, 0, 255], 3)
        cv2.line(warp_out, ( int(dst[2][0]), int(dst[2][1] ) ), ( int(dst[3][0]), int(dst[3][1] ) ), [0, 0, 255], 3)
        cv2.line(warp_out, ( int(dst[3][0]), int(dst[3][1] ) ), ( int(dst[0][0]), int(dst[0][1] ) ), [0, 0, 255], 3)
        if warp_out.shape[2]!=1:
            cv2.imwrite( warp_out_fname, warp_out )
        else:
            cv2.imwrite( warp_out_fname, warp_out, cmap=cm.gray )

    return (warped, M, Minv)
