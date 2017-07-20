#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

import sys
from masters_project.msg import flow_vectors, flow_vectors_list, coordinate


class Optic_Flow:
    def __init__(self):

        self.bridge = CvBridge()
        self.pub = rospy.Publisher('optic_flow_parameters', flow_vectors_list, queue_size=100)


        self.show_hsv = False
        self.cv_image = None
        self.once = True
        self.blerg = False
        self.prevgray = None
        self.count = 0

        # rospy.Rate(20)
        rospy.init_node('optic_flow', anonymous=True)

        rospy.Subscriber('camera_image', Image, self.callback)
        # rate = rospy.Rate(5)
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

    # This method is called whenever the node receives a msg
    # from its subscriber
    def callback(self, ros_image):
        try:
            # Print out image height and width for testing
            height = ros_image.height
            print height
            width = ros_image.width
            print width
            # convert ros image back to cv format to compute optical flow
            self.cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")

            # Also for doing classification by hand
            # cv2.imwrite(str(self.count)+".png", self.cv_image) # testing to make sure no frames are published twice.
            # Get around the right number of frames but it varies a bit so most be sending some repeats! Maybe to do
            # with rospy.rate
            self.count += 1
            # For the first image we receive convert it to gray and set it as prev so that we can compute optical flow from next image and prev one
            if self.cv_image is not None and self.once:
                self.prevgray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
                self.blerg = True

            # Check node has received image so doesn't do computation too early/don't enter this loop when we only have
            # one image as can't compute flow
            if self.cv_image is not None and not self.once and self.blerg:

                # Get the next image and convert it to grayscale
                gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

                # Computes a dense optical flow using the Gunnar Farneback's algorithm
                # returns a 2-channel image with X and Y magnitudes and orientation
                flow = cv2.calcOpticalFlowFarneback(self.prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # Set current image to previous
                self.prevgray = gray
                #fx, fy = flow[:, :, 0], flow[:, :, 1]
                step = 16

                h, w = self.cv_image.shape[:2]
                y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
                fx, fy = flow[y, x].T
                lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
                lines = np.int32(lines + 0.5)

                print "!!!!" + str(len(lines)) + " " + str(len(lines[0])) + " " + str(len(lines[0][0]))

                print lines
                #print "!!!!!rows" + str(len(fx)) + "columns" + str(len(fx[0]))
                start_points = coordinate()
                end_points = coordinate()
                temp = flow_vectors()
                msg = flow_vectors_list()

                """
                # Iterate over flow vector array of image and add each one to the custom message
                for (x, y) in self.matrix_iterator(fx, fy):
                    temp.flow_vectors[0] = x
                    temp.flow_vectors[1] = y
                    msg.parameters += [temp]
                """

                for i in range(len(lines)):
                    start_points.coordinates[0] = lines[i][0][0]
                    start_points.coordinates[1] = lines[i][0][1]
                    end_points.coordinates[0] = lines[i][1][0]
                    end_points.coordinates[1] = lines[i][1][1]
                    temp.flow_vectors[0] = start_points
                    temp.flow_vectors[1] = end_points
                    msg.parameters += [temp]

                # Don't think I need these anymore this was when I thought I had to reassemble list into image shape
                msg.height = height # when printing out height in machine learning node it is 0 WHY?????
                msg.width = width # width is 640 ~160

                # publish flow vectors message
                self.pub.publish(msg)
            # make sure loop is only entered once    
            self.once = False
            
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

    """
    Helper method to iterate over the xflow and yflow matrices at once
    """
    def matrix_iterator(self, x_matrix, y_matrix):
        for i in range(len(x_matrix)):
            for j in range(len(x_matrix[i])):
                yield(x_matrix[i][j], y_matrix[i][j])




if __name__ == '__main__':
    try:
        flowObj = Optic_Flow()
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'Shutting down'

