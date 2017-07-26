#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


"""
Made a node to save images so that classification of webcam streams is easier
and so that the images saved will be the same as the frames sent by image_publisher
node (as the node drops quite a lot of frames)

"""
class Image_Saver:
    def __init__(self):

        self.bridge = CvBridge()

        self.cv_image = None
        self.count = 0

        rospy.init_node('image_saver', anonymous=True)

        rospy.Subscriber('camera_image', Image, self.callback)


    # This method is called whenever the node receives a msg
    # from its subscriber
    def callback(self, ros_image):
        try:

            # convert ros image back to cv format to compute optical flow
            self.cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")

            # Also for doing classification by hand
            cv2.imwrite(str(self.count)+".png", self.cv_image) # testing to make sure no frames are published twice.
            # Get around the right number of frames but it varies a bit so most be sending some repeats! Maybe to do
            # with rospy.rate
            self.count += 1

        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    try:
        flowObj = Image_Saver()
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'Shutting down'
