#!/usr/bin/env python

PKG = 'masters_project' #??? is this right
NAME = "test_image_publisher"
import rospy
import unittest
from sensor_msgs.msg import Image
import rostest

class TestImagePublisher(unittest.TestCase):

    # could test for invalid input

    def __init__(self, *args):

        super(TestImagePublisher, self).__init__(*args)
        self.passed = False
        rospy.init_node(NAME)

# ?????????
 #   def setUp(self):
 #       rospy.init_node("test_image_publisher")

    def subscribe(self, data):
        if data:
            if data.height > 0:
                self.passed = True

    def test_image_publish(self):
        rospy.Subscriber("camera_image", Image, self.subscribe)

        self.assertTrue(self.passed, "Successfully received Image from image_publisher: " + str(self.passed))


if __name__ == '__main__':
    rostest.rosrun(PKG, NAME, TestImagePublisher)