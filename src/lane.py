import numpy as np

# A class representing a lane
class Lane():
    # constructor. Create a Lane representation
    # laneid: the id (name) of the lane. It can be 'left', or 'right'
    # lane_width: lane width in meters
    def __init__(self, laneid, lane_width):
        self.laneid = laneid
        self.lane_width = lane_width
        self.detected = False
        self.current_fit = [np.array([False])]
        self.allx = None
        self.ally = None
        # difference in fit coefficients between last and new fits
        self.diffs = None
        self.maximum_not_confident_frames = 12                          # if the lane was not detected since this many frames, we turn back to the full screen histogram + sliding windows method. Default value: About half sec.
        self.undetected_since_frame = 0                                 # the last time the lane was undetected
        self.min_number_of_activated_points_op_top_part = 6             # if there's not this many activated points in the upper part of the screen, we don't use the polyfit algorithm on it
        self.top_part_on_warped_image = 360
        self.too_many_activated_pixels = 40.0                           # percentage
        self.too_many_activated_pixels_when_areasearch = 90.0           # percentage where there's a previously fitted polyline
        self.curverad = 0

    # Did we lost the lane? If its return value is True, then we need to fall back to the histogram/sliding windows lane detection mode.
    def lost(self, frame_num):
        if frame_num == 1:
            # first frame: there's no previous polyfit
            return True

        if self.detected is False:
            # If we could not detect the lane in the last maximum_not_confident_frames frames, we need to fallback
            if (frame_num-self.undetected_since_frame) > self.maximum_not_confident_frames:
                return True

        # otherwise, we could detect the lane OR we can still use the previous valid lane detection's result
        return False

    # A new polynomial coefficient set was calculated. Store it internally.
    def setcurrentfit(self, newfit, binary_warped, confident, frame_num, reason = ''):

        if (self.detected is True) and (confident is False):
            print("Frame {}: lane {} was lost. Reason: {}".format(frame_num, self.laneid, reason))
        if (self.detected is False) and (confident is True):
            print("Frame {}: lane {} was recognized.".format(frame_num, self.laneid))

        if (newfit[0] != False):
            if self.current_fit[0] != False:
                self.diffs = newfit - self.current_fit
            self.current_fit = newfit
            y_bottom = binary_warped.shape[0] - 1       # y value where we want radius of curvature (bottom of image)
            ym_per_pix = 30 / 720                       # meters per pixel in y dimension
            xm_per_pix = self.lane_width / 700          # meters per pixel in x dimension

            # convert parabol coefficients from pixel based to meters based.
            fit_cr = ( newfit[0]* xm_per_pix / (ym_per_pix ** 2), newfit[1]*xm_per_pix / ym_per_pix, newfit[2] )
            # calculate the radius in meters
            self.curverad = pow( (1+(2*fit_cr[0]*y_bottom+fit_cr[1])**2), 1.5) / (abs(2*fit_cr[0]))

        if (self.current_fit[0] != False):
            # calculate allx and ally is we have a poly fit
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0], dtype=np.int32)
            fitx = self.current_fit[0] * ploty ** 2 + self.current_fit[1] * ploty + self.current_fit[2]
            self.allx = fitx.astype(int)
            self.ally = ploty

        if (self.detected is True) & (confident is False):
            self.undetected_since_frame = frame_num
        self.detected = confident

    # Sliding window/histogram algorithm ending part: try to fit the lane to the given activated points is possible
    def fit_poly(self, lane_inds, nonzerox, nonzeroy, binary_warped, frame_num, margin = 50):
        if (frame_num>1) and (self.current_fit[0] != False):
            f = self.current_fit
            toppartinds = (nonzeroy < self.top_part_on_warped_image) & ((nonzerox > (f[0] * (nonzeroy ** 2) + f[1] * nonzeroy +
                f[2] - margin)) & (nonzerox < (f[0] * (nonzeroy ** 2) +
                f[1] * nonzeroy + f[2] + margin)) )

            topxpts = nonzerox[toppartinds]
            topypts = nonzeroy[toppartinds]

            if len(topxpts) < self.min_number_of_activated_points_op_top_part:
                self.setcurrentfit(self.current_fit, binary_warped, False, frame_num, 'Too many activated points at the top part of the image')
                return
        xpts = nonzerox[lane_inds]
        ypts = nonzeroy[lane_inds]

        fit = np.polyfit(ypts, xpts, 2)
        self.setcurrentfit( fit, binary_warped, True, frame_num )

    # Lane detection using the previously detected lane curve
    def fit_poly_use_poly_area(self, nonzerox, nonzeroy, binary_warped, frame_num, margin = 50):
        f = self.current_fit

        if f[0] == False:
            # there's no previous recognized lane
            return

        lane_inds = ((nonzerox > (f[0] * (nonzeroy ** 2) + f[1] * nonzeroy +
            f[2] - margin)) & (nonzerox < (f[0] * (nonzeroy ** 2) +
            f[1] * nonzeroy + f[2] + margin)))

        s = 2 * margin *  binary_warped.shape[0] // 2
        pcnt = 100.0 * len(lane_inds) / s
        # print("Win {} lpcnt {} rpcnt {}".format(window, lactpcnt, ractpcnt))
        if pcnt > self.too_many_activated_pixels_when_areasearch:
            self.setcurrentfit(self.current_fit, binary_warped, False, frame_num, 'Too many activated points around the previously detected lanes')
        else:
            self.fit_poly(lane_inds, nonzerox, nonzeroy, binary_warped, frame_num)
