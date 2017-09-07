#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
from masters_project.msg import flow_vectors
from helpers import convert_ROS_to_CV


class OpticFlow:
    """
    ROS node that receives an image and computes the optic flow for it. Then it creates
    a subsampled version of the optic flow and publishes it to the optic_flow_parameters
    topic.
    """
    def __init__(self):
        self.pub = rospy.Publisher('optic_flow_parameters', flow_vectors, queue_size=100)
        self.cv_image = None  # stores the image in openCV format
        self.prevgray_initialised = False  # Boolean to check prevray is only initialised once
        self.prevgray = None  # stores previous grayscale image for optic flow
        rospy.init_node('optic_flow', anonymous=True)
        rospy.Subscriber('camera_image', Image, self.callback)

    # This is called whenever a message is published to camera_image topic
    def callback(self, ros_image):
        # convert ros image back to cv format to compute optical flow
        self.cv_image = convert_ROS_to_CV(ros_image, "passthrough")
        # For the first image we receive convert it to gray and set it as prev so that we can compute optical
        # flow from next image and prev one
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
            # Set prevgray to current image
            self.prevgray = gray
            msg = flow_vectors()
            flow = self.sub_sample_flow_vectors(flow, 21)
            flow_array = np.array(flow)
            flow = flow_array.tolist()
            for i in range(len(flow)):
                    msg.flow_vectors += flow[i]
            # publish flow vectors message
            self.pub.publish(msg)

    # helper method to average flow vectors so that each step x step group
    # of vectors become one average one.
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


if __name__ == '__main__':
    try:
        flowObj = OpticFlow()
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'Shutting down'

