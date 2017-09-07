#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
from helpers import convert_ROS_to_CV



class Image_Saver:
    """
    ROS node that saves images as PNG files so that the classification of webcam streams is easier
    and so that the images saved will be the same as the frames sent by image_publisher
    node.
    """
    def __init__(self):
        self.bridge = CvBridge()
        self.cv_image = None
        self.count = 0
        rospy.init_node('image_saver', anonymous=True)
        rospy.Subscriber('camera_image', Image, self.callback)

    # Converts each image to an OpenCV format and saves it
    def callback(self, ros_image):
        # convert ros image back to cv format to compute optical flow
        self.cv_image = convert_ROS_to_CV(ros_image, desired_encoding="passthrough")
        # Save the image
        cv2.imwrite(str(self.count)+".png", self.cv_image) # testing to make sure no frames are published twice.
        self.count += 1


if __name__ == '__main__':
    try:
        flowObj = Image_Saver()
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'Shutting down'
