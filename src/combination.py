import sys
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import pickle
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import cvutils

global blurred, gradx_binary, grady_binary, mag_binary, dir_binary, s_binary, l_binary, gray_binary
global grayimg

app = QtGui.QApplication([])
mtx = None
dist = None
blurred = None
gradx_binary = grady_binary=mag_binary=dir_binary=s_binary=l_binary=gray_binary=None
grayimg = None

def undistort_image(img):
    global mtx, dist
    if mtx is None:
        print("Loading camera calibration parameters.")
        s = pickle.load(open(cvutils.calfilename, "rb"))
        mtx = s['mtx']
        dist = s['dist']
    return cvutils.undistort_image(img, mtx, dist)

def process_frame(dumpoutput):
    global grayimg, combination_image_widget, latest_binary_widget, ghls
    global blurred, gradx_binary, grady_binary, mag_binary, dir_binary, s_binary, l_binary, gray_binary

    params = [
        #{'name': 'imagefile_name', 'type': 'str', 'value': 'original41.jpg'},
        {'name': 'imagefile_name', 'type': 'str', 'value': '../test_images/test1.jpg'},
        #{'name': 'imagefile_name', 'type': 'str', 'value': '../test_images/straight_lines2.jpg'},
        {'name': 'kernel_size', 'type': 'int', 'value': 3},
        {'name': 'mag_kernel_size', 'type': 'int', 'value': 7},
        {'name': 'dir_kernel_size', 'type': 'int', 'value': 7},                 # this is not used anymore
        {'name': 'gradx_min', 'type': 'int', 'value': 65},
        {'name': 'gradx_max', 'type': 'int', 'value': 255},
        {'name': 'grady_min', 'type': 'int', 'value': 100},
        {'name': 'grady_max', 'type': 'int', 'value': 255},
        {'name': 'mag_min', 'type': 'int', 'value': 128},
        {'name': 'mag_max', 'type': 'int', 'value': 255},
        {'name': 'gray_min', 'type': 'int', 'value': 200 },
        {'name': 'gray_max', 'type': 'int', 'value': 199 },
        {'name': 'dir_min', 'type': 'float', 'value': 0.4, 'step':0.05 },       # this is not used anymore
        {'name': 'dir_max', 'type': 'float', 'value': np.pi/2, 'step':0.05 },   # this is not used anymore
        {'name': 's_min', 'type': 'int', 'value': 140},
        {'name': 's_max', 'type': 'int', 'value': 255},
        {'name': 'l_min', 'type': 'int', 'value': 220},
        {'name': 'l_max', 'type': 'int', 'value': 255},
        ]
    pickle.dump({ 'params': params }, open(cvutils.paramfilename, "wb"))
    param = Parameter.create(name='params', type='group', children=params)
    img = cv2.imread(param['imagefile_name'])
    assert img is not None, "Input frame is empty ! File not found or read error."

    undistorted = undistort_image(img)
    grayimg = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    ghls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)

    def showpic(img, isgrayscaleimage = False):
        global latest_binary_widget
        if isgrayscaleimage is False:
            gpic = img * 255
        else:
            gpic = img
        latest_binary_widget.setPixmap(QtGui.QPixmap(QtGui.QImage(gpic.data, gpic.shape[1], gpic.shape[0], gpic.shape[1], QtGui.QImage.Format_Grayscale8)))

    def change(param, changes):
        global grayimg, ghls, combination_image_widget, latest_binary_widget
        global blurred, gradx_binary, grady_binary, mag_binary, dir_binary, s_binary, l_binary, gray_binary

        varname = changes[0][0].name()
        if varname=='gradx_min' or varname=='gradx_max':
            gradx_binary = cvutils.abs_sobel_thresh(blurred, orient='x', thresh_min=param['gradx_min'], thresh_max=param['gradx_max'])
            showpic(gradx_binary)
        if varname=='grady_min' or varname=='grady_max':
            grady_binary = cvutils.abs_sobel_thresh(blurred, orient='y', thresh_min=param['grady_min'], thresh_max=param['grady_max'])
            showpic(grady_binary)
        if varname=='mag_min' or varname=='mag_max' or varname=='mag_kernel_size':
            mag_binary = cvutils.mag_thresh(blurred, sobel_kernel=param['mag_kernel_size'], mag_thresh=(param['mag_min'], param['mag_max']))
            showpic(mag_binary)
        if varname=='dir_min' or varname=='dir_max' or varname=='gray_min' or varname=='gray_max' or varname=='dir_kernel_size':
            #dir_binary = dir_threshold(blurred, sobel_kernel=param['dir_kernel_size'], thresh=(param['dir_min'], param['dir_max']))
            gray_binary = cvutils.gray_thresh(blurred, thresh=(param['gray_min'], param['gray_max']))
            showpic(gray_binary)
        if varname=='kernel_size':
            blurred = cv2.GaussianBlur(grayimg, (param['kernel_size'], param['kernel_size']), 0)
            gradx_binary = cvutils.abs_sobel_thresh(blurred, orient='x', thresh_min=param['gradx_min'], thresh_max=param['gradx_max'])
            grady_binary = cvutils.abs_sobel_thresh(blurred, orient='y', thresh_min=param['grady_min'], thresh_max=param['grady_max'])
            mag_binary = cvutils.mag_thresh(blurred, sobel_kernel=param['mag_kernel_size'], mag_thresh=(param['mag_min'], param['mag_max']))
            gray_binary = cvutils.gray_thresh(blurred, thresh=(param['gray_min'], param['gray_max']))
            showpic(blurred)
        if varname=='s_min' or varname=='s_max':
            s_binary = cvutils.hls_s_select(ghls, thresh=(param['s_min'], param['s_max']))
            showpic(s_binary)
        if varname=='l_min' or varname=='l_max':
            l_binary = cvutils.hls_l_select(ghls, thresh=(param['l_min'], param['l_max']))
            showpic(l_binary)
        if varname=='imagefile_name':
            img = cv2.imread(param['imagefile_name'])
            assert img is not None, "Input frame is empty ! File not found or read error."
            undistorted = undistort_image(img)
            grayimg = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
            ghls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
            (blurred, gray_binary, gradx_binary, grady_binary, mag_binary, s_binary, l_binary) = cvutils.gen_all_binary_images( grayimg, ghls, param )
            showpic(grayimg, True)

        combined = cvutils.combine_images(gray_binary, gradx_binary, grady_binary, mag_binary, s_binary, l_binary) * 255
        pix = QtGui.QPixmap(QtGui.QImage(combined.data, combined.shape[1], combined.shape[0], combined.shape[1], QtGui.QImage.Format_Grayscale8))
        combination_image_widget.setPixmap(pix)

    param.sigTreeStateChanged.connect(change)

    (blurred, gray_binary, gradx_binary, grady_binary, mag_binary, s_binary, l_binary ) = cvutils.gen_all_binary_images( grayimg, ghls, param )
    combined = cvutils.combine_images(gray_binary, gradx_binary, grady_binary, mag_binary, s_binary, l_binary) * 255

    if dumpoutput is True:
        warped = cvutils.warper( undistorted,
                                 '../output_images/unwarped_{}'.format(os.path.basename(param['imagefile_name']) ),
                                 '../output_images/warped_{}'.format(os.path.basename(param['imagefile_name']) )
                                 )
        plt.imsave('../output_images/combined_{}'.format(os.path.basename(param['imagefile_name'])), combined, cmap=cm.gray);

    else:
        t = ParameterTree()
        t.setParameters(param, showTop=True)
        combination_window = QtGui.QWidget()
        latest_changed_binary_window = QtGui.QWidget()
        latest_changed_binary_window.setWindowTitle("Latest modified parameter's image")

        layout = QtGui.QGridLayout()
        combination_window.setLayout(layout)
        layout.addWidget(t, 0, 0, 1, 1)

        latest_binary_widget=QtGui.QLabel(latest_changed_binary_window)
        pix = QtGui.QPixmap(QtGui.QImage(grayimg.data, grayimg.shape[1], grayimg.shape[0], grayimg.shape[1], QtGui.QImage.Format_Grayscale8))
        latest_binary_widget.setPixmap(pix)
        latest_changed_binary_window.show()
        latest_binary_widget.setGeometry(QtCore.QRect(10, 40, pix.width(), pix.height()))
        latest_changed_binary_window.resize(pix.width()+20, pix.height()+100)

        combination_image_widget=QtGui.QLabel("")
        combination_image_widget.setPixmap(QtGui.QPixmap(QtGui.QImage(combined.data, combined.shape[1], combined.shape[0], combined.shape[1], QtGui.QImage.Format_Grayscale8)))
        layout.addWidget(combination_image_widget, 0, 1, 1, 1)
        combination_window.show()
        combination_window.resize(1680, 768)
        QtGui.QApplication.instance().exec_()

if __name__ == '__main__':

    dumpoutput = False               # if this is True, the test image's binary combined image output will just be written to the ../output_images/ directory as "combined_{filename}", and the application will exit.
                                    # in case of False, the QT application will start

    process_frame(dumpoutput)
