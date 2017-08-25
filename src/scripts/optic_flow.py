#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
from masters_project.msg import flow_vectors


class OpticFlow:
    def __init__(self):

        # create CvBridge object to use functionality to convert between ROS and OpenCV images
        self.pub = rospy.Publisher('optic_flow_parameters', flow_vectors, queue_size=100)
        self.cv_image = None  # stores the image in openCV format
        self.once = True  # Boolean to make sure conditions are entered or not entered on first execution (think this is whack code)
        self.prevgray_initialised = False  # Boolean to check conditions also
        self.prevgray = None  # stores previous grayscale image for optic flow
        self.display_image = None
        rospy.init_node('optic_flow', anonymous=True)

        rospy.Subscriber('camera_image', Image, self.callback)

    # This method is called whenever the node receives a msg
    # from its subscriber
    def callback(self, ros_image):
        try:
            # Print out image height and width for testing
            height = ros_image.height
            width = ros_image.width
            print height, width
            # convert ros image back to cv format to compute optical flow
            self.cv_image = self.convert_ROS_to_CV(ros_image, "passthrough")
            # For the first image we receive convert it to gray and set it as prev so that we can compute optical
            #  flow from next image and prev one


            # WHY ARE THE SELF.CV_IMAGE NOT NONE CHECKS NECESSARY? CHECK IT WORKS WITHOUT

            if self.cv_image is not None and not self.prevgray_initialised:
                self.prevgray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
                self.prevgray_initialised = True

            # Check node has received image so doesn't do computation too early/don't enter this loop when we only have
            # one image as can't compute flow
            if self.cv_image is not None and self.prevgray_initialised:

                # Get the next image and convert it to grayscale
                gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
                # Computes a dense optical flow using the Gunnar Farneback's algorithm
                # returns a 2-channel image with X and Y magnitudes and orientation
                flow = cv2.calcOpticalFlowFarneback(self.prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                f, g, h = flow.shape
                print "Y is " + str(f) + " X is " + str(g) + " Z is " + str(h)
                # Set prevgray to current image
                self.prevgray = gray


                # bit that computes the optic flow lines - repeated in other method - improve so don't have duplicate code

                msg = flow_vectors()


                flow = self.sub_sample_flow_vectors(flow, 21)
                blerg = np.array(flow)
                flow = blerg.tolist()

                for i in range(len(flow)):
                        msg.flow_vectors += flow[i]


                print msg.flow_vectors
                # Don't think I need these anymore this was when I thought I had to reassemble list into image shape
                #msg.height = height # when printing out height in machine learning node it is 0 WHY?????
                #msg.width = width # width is 640 ~160

                # publish flow vectors message
                self.pub.publish(msg)
            # make sure loop is only entered once    
            self.once = False
        except CvBridgeError as e:
            print(e)


    def sub_sample_flow_vectors(self, flow, step):
        sample = []
        for i in range(0, len(flow)-step, step):
            for j in range(0, len(flow[i])-step, step):
                sumX = 0
                sumY = 0
                for s in range(step):
                    for t in range(step):
                        sumX += flow[i+s][j+t][0]
                        sumY += flow[i+s][j+t][1]

                averageX = sumX / (step*step)
                averageY = sumY / (step*step)
                sample += [[averageX, averageY]]

        return sample

    def convert_ROS_to_CV(self, image, encoding):
        try:
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(image, desired_encoding=encoding)
            return cv_image
        except CvBridgeError, e:
            print e


if __name__ == '__main__':
    try:
        flowObj = OpticFlow()
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'Shutting down'

