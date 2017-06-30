#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2


class Optic_Flow():

    def callback(self, ros_image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

    def draw_flow(self, img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    def draw_hsv(self, flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx * fx + fy * fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v * 4, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def __init__(self):

        self.bridge = CvBridge()

        self.show_hsv = False
        self.cv_image = None


        rospy.init_node('optic_flow', anonymous=True)

        rospy.Subscriber('camera_image', Image, self.callback)

        # while self.cv_image != None:

        #     ret, prev = self.cv_image
        #     prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        #     show_hsv = False
        #     ret, img = self.cv_image
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #     prevgray = gray

        #     cv2.imshow('flow', self.draw_flow(gray, flow))
        #     if show_hsv:
        #         cv2.imshow('flow HSV', self.draw_hsv(flow))

        #     ch = cv2.waitKey(5)
        #     if ch == 27:
        #         break
        #     if ch == ord('1'):
        #         show_hsv = not show_hsv
        #         print('HSV flow visualization is', ['off', 'on'][show_hsv])

        # cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        flowObj = Optic_Flow()
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'Shutting down'

