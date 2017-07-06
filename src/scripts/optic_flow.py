#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

import sys
#sys.path.insert(0. ../../msg)
from masters_project.msg import flow_vectors, flow_vectors_list


class Optic_Flow():

    def callback(self, ros_image):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
            if self.cv_image != None:
                print 'ENTERED'
                prev = self.cv_image
                prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

                # returns a 2-channel image with X and Y magnitudes and orientation
                flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                fx, fy = flow[:, :, 0], flow[:, :, 1]
                temp = flow_vectors()
                msg = flow_vectors_list()
                '''fx_list = []
                for cell in fx.flat:
                    print cell
                    temp.flow_vectors[0] = cell
                    msg.parameters[i] = temp

                                print fx
                temp = flow_vectors()
                msg = flow_vectors_list()
                for i in range(fx):
                    temp.flow_vectors[0] = fx[i]
                    temp.flow_vectors[1] = fy[i]
                    msg.parameters[i] = temp'''
               
                for (x, y) in self.matrix_iterator(fx, fy):
                    temp.flow_vectors[0] = x
                    temp.flow_vectors[1] = y
                    msg.parameters += [temp]






                self.pub.publish(msg)
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

    def matrix_iterator(self, x_matrix, y_matrix):
        for i in range(len(x_matrix)):
            for j in range(len(x_matrix[i])):
                yield(x_matrix[i][j], y_matrix[i][j])


    def __init__(self):

        self.bridge = CvBridge()
        self.pub = rospy.Publisher('optic_flow_parameters', flow_vectors_list, queue_size=10)

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

