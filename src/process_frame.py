import sys
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import pickle
import pyqtgraph as pg
from moviepy.editor import VideoFileClip
import lane
import car

def process_frame(img):
    global car

    car.newframe()

    # process 1 frame

    if car.debuglevel > 0:
        plt.imsave("original" + str(car.frame_number) + ".jpg", img);

    car.process_frame_binary(img)
    car.warper(car.binary_image)
    if car.debuglevel > 0:
        plt.imsave("warped" + str(car.frame_number) + ".jpg", car.warped_image, cmap=cm.gray);

    if car.left_lane.lost(car.frame_number) or car.right_lane.lost(car.frame_number):
        car.find_lanes()
    else:
        car.search_around_poly()
    # ---------------------------
    car.calc_car_position()
    # ---------------------------------
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(car.warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    if (car.left_lane.allx is not None) and (car.right_lane.ally is not None):
        pts_left = np.array([np.transpose(np.vstack([car.left_lane.allx, car.left_lane.ally]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([car.right_lane.allx, car.right_lane.ally])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        newwarp = cv2.warpPerspective(color_warp, car.Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(car.undistorted_image, 1, newwarp, 0.3, 0)
    else:
        result = car.gray_image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText( result, 'Radius of curvature = {} m'.format((car.left_lane.curverad + car.right_lane.curverad) // 2),
                (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA )
    if car.distance_from_lane_center < 0:
        txt = 'left'
    else:
        txt = 'right'

    cv2.putText(result, 'Vehicle is {:.2f}m {} of center'.format(abs(car.distance_from_lane_center), txt ),
                (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if car.debuglevel>0:
        plt.imsave("frame"+str(car.frame_number)+".jpg", result, cmap = cm.gray);
    return result

if __name__ == '__main__':

    debuglevel = 0

    car = car.Car( debuglevel )

    #if debuglevel == 1:
    #    img = cv2.imread('original38.jpg')
    #    process_frame(img)

    fname = 'project_video'
    output_video_path = '../videos_output/{}.mp4'.format(fname)
    if car.debuglevel>0:
        f=0
        t=1
        output_video_path = '{}{}-{}.mp4'.format(fname, f, t)
        mv = VideoFileClip('../{}.mp4'.format(fname)).subclip(f, t)
    else:
        mv = VideoFileClip('../{}.mp4'.format(fname))
    clip = mv.fl_image(process_frame)
    clip.write_videofile(output_video_path, audio=False)

