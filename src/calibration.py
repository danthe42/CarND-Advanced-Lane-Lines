import sys
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
import pickle

calfilename = "cal_result.p"

def calibratecamera(dirname, outputdirname):
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = []
    imgpoints = []
    img_w = 0
    img_h = 0
    # Make a list of calibration images
    images = os.listdir(dirname)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(dirname + '/' + fname)
        if img_w==0:
            img_w = img.shape[1]
            img_h = img.shape[0]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            img2 = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imwrite( outputdirname+'/'+fname , img2 )
        else:
            print('Notice: Skipping image %s, as it does not display the full chessboard.'%(fname))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img_w, img_h), None, None)

    # Serialize camera calibration results in a persistent way, to be used later
    pickle.dump({ 'mtx': mtx, 'dist': dist }, open(calfilename, "wb"))

if __name__ == '__main__':
    calibratecamera('../camera_cal', '../cal_image_output')
    print('Calibration done, result written to file: %s'%(calfilename))
