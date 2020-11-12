import pickle
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import cv2
import cvutils
import lane
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

# A class, containing all data what is necessary for processing the current frame, and find the lanes, curvature, radius, position, etc. in the project
class Car():
    def newframe(self):
        self.binary_image = None
        self.undistorted_image = None
        self.gray_image = None
        self.hls_image = None
        self.warped_image = None
        self.distance_from_lane_center = 0
        self.frame_number = self.frame_number + 1

    # debuglevel:
    #   0 =  No DEBUG
    #   1 = Dump frame images
    #   2 = Debuglevel 1 and draw find_lanes results into resulting warped image
    def __init__(self, debuglevel = 0):
        self.lane_width = 3.7                           # constant
        self.right_lane = lane.Lane('right', self.lane_width)
        self.left_lane = lane.Lane('left', self.lane_width)
        self.frame_number = 0
        self.debuglevel = debuglevel
        s = pickle.load(open(cvutils.calfilename, "rb"))
        self._calib_mtx = s['mtx']
        self._calib_dist = s['dist']
        s = pickle.load(open(cvutils.paramfilename, "rb"))
        self.param = Parameter.create(name='params', type='group', children=s['params'])
        self.M = None
        self.Minv = None

    def calc_car_position(self):
        if (self.left_lane.current_fit[0] != False) & (self.right_lane.current_fit[0] != False):
            ploty = self.binary_image.shape[0]-1
            xl = self.left_lane.current_fit[0] * ploty ** 2 + self.left_lane.current_fit[1] * ploty + self.left_lane.current_fit[2]
            xr = self.right_lane.current_fit[0] * ploty ** 2 + self.right_lane.current_fit[1] * ploty + self.right_lane.current_fit[2]
            self.distance_from_lane_center = self.lane_width * ( self.binary_image.shape[1] / 2 - (xl+xr) // 2 ) / (xr-xl)


    def process_frame_binary(self, camera_raw_img):
        self.undistorted_image = cvutils.undistort_image(camera_raw_img, self._calib_mtx, self._calib_dist)
        if self.debuglevel>1:
            plt.imsave("undistorted" + str(self.frame_number) + ".jpg", self.undistorted_image);

        self.gray_image = cv2.cvtColor(self.undistorted_image, cv2.COLOR_RGB2GRAY)
        self.hls_image = cv2.cvtColor(self.undistorted_image, cv2.COLOR_RGB2HLS)

        param = self.param
        blurred = cv2.GaussianBlur(self.gray_image, (param['kernel_size'], param['kernel_size']), 0)
        gradx_binary = cvutils.abs_sobel_thresh(blurred, orient='x', thresh_min=param['gradx_min'], thresh_max=param['gradx_max'])
        grady_binary = cvutils.abs_sobel_thresh(blurred, orient='y', thresh_min=param['grady_min'], thresh_max=param['grady_max'])
        mag_binary = cvutils.mag_thresh(blurred, sobel_kernel=param['mag_kernel_size'],
                                mag_thresh=(param['mag_min'], param['mag_max']))
        dir_binary = cvutils.gray_thresh(blurred, thresh=(param['gray_min'], param['gray_max']))
        s_binary = cvutils.hls_s_select(self.hls_image, thresh=(param['s_min'], param['s_max']))
        l_binary = cvutils.hls_l_select(self.hls_image, thresh=(param['l_min'], param['l_max']))

        combined = np.zeros_like(self.gray_image)
        combined[((dir_binary == 1) | (gradx_binary == 1) | (mag_binary == 1) | (s_binary == 1) | (l_binary == 1)) & (
                    grady_binary == 0)] = 1
        combined = combined * 255
        self.binary_image = combined
        if self.debuglevel>0:
            plt.imsave("unwarped" + str(self.frame_number) + ".jpg", self.binary_image, cmap=cm.gray);
            plt.imsave("gray" + str(self.frame_number) + ".jpg", self.gray_image, cmap=cm.gray);

    def warper(self, img):
        (self.warped_image, self.M, self.Minv) = cvutils.warper(img)

    # margin parameter: The width of the margin around the previous window (from the previous frame) to search
    def search_around_poly(self):

        #histogram = np.sum(self.warped_image[self.warped_image.shape[0] // 2:, :], axis=0)
        #if self.frame_number == 36:
            #cv2.imshow('lanes{}'.format(self.frame_number), self.warped_image)
            #print(str(histogram))
        # Grab activated pixels
        nonzero = self.warped_image.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])
        self.left_lane.fit_poly_use_poly_area(nonzerox, nonzeroy, self.binary_image, self.frame_number)
        self.right_lane.fit_poly_use_poly_area(nonzerox, nonzeroy, self.binary_image, self.frame_number)

    def find_lanes(self, nwindows = 9, margin = 100, minpix = 50):
        binary_warped = self.warped_image
        histogram = np.sum(self.warped_image[self.warped_image.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        if self.debuglevel>1:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            ### Find the four below boundaries of the window ###
            win_xleft_low = max(0, leftx_current - margin)
            win_xleft_high = min(binary_warped.shape[1] - 1, leftx_current + margin)
            win_xright_low = max(0, rightx_current - margin)
            win_xright_high = min(binary_warped.shape[1] - 1, rightx_current + margin)
            # Draw the windows on the visualization image
            if self.debuglevel > 1:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left_inds = np.where(np.logical_and(np.logical_and(nonzerox > win_xleft_low, nonzerox < win_xleft_high),
                                                     np.logical_and(nonzeroy > win_y_low, nonzeroy <= win_y_high)))[0]
            good_right_inds = np.where(np.logical_and(np.logical_and(nonzerox > win_xright_low, nonzerox < win_xright_high),
                                                      np.logical_and(nonzeroy > win_y_low, nonzeroy <= win_y_high)))[0]
            lsum = ( win_xleft_high-win_xleft_low - 1) * ( win_y_high - win_y_low - 1 )
            rsum = ( win_xright_high-win_xright_low - 1) * ( win_y_high - win_y_low - 1 )
            lactpcnt = 100.0 * len(good_left_inds) / lsum
            ractpcnt = 100.0 * len(good_right_inds) / rsum
            #print("Win {} lpcnt {} rpcnt {}".format(window, lactpcnt, ractpcnt))
            if (lactpcnt > self.left_lane.too_many_activated_pixels) or (ractpcnt > self.right_lane.too_many_activated_pixels):
                self.left_lane.setcurrentfit(self.left_lane.current_fit, binary_warped, False, self.frame_number, 'Too many activated points in one of the windows during sliding windows algorithm')
                self.right_lane.setcurrentfit(self.right_lane.current_fit, binary_warped, False, self.frame_number, 'Too many activated points in one of the windows during sliding windows algorithm')
                return

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### If we found > minpix pixels, recenter next window ###
            if len(good_left_inds) >= minpix:
                leftx_current = int(round(np.mean(nonzerox[good_left_inds])))
            if len(good_right_inds) >= minpix:
                rightx_current = int(round(np.mean(nonzerox[good_right_inds])))
            ### (`right` or `leftx_current`) on their mean position ###

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        self.left_lane.fit_poly(left_lane_inds, nonzerox, nonzeroy, binary_warped, self.frame_number)
        self.right_lane.fit_poly(right_lane_inds, nonzerox, nonzeroy, binary_warped, self.frame_number)

        if self.debuglevel>1:
            xpts = nonzerox[left_lane_inds]
            ypts = nonzeroy[left_lane_inds]
            out_img[ypts, xpts] = [255, 0, 0]
            xpts = nonzerox[right_lane_inds]
            ypts = nonzeroy[right_lane_inds]
            out_img[ypts, xpts] = [0, 0, 255]

            out_img[self.left_lane.ally, self.left_lane.allx] = [0, 255, 255]
            out_img[self.right_lane.ally, self.right_lane.allx] = [0, 255, 255]
            plt.imsave("../output_images/cdolor_fit_lines.jpg", out_img);
            #cv2.imshow('warpedimg{}'.format(self.frame_number), out_img)
            cv2.waitKey(500000)
#            cv2.destroyAllWindows()
