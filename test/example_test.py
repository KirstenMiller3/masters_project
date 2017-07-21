#!/usr/bin/env python

PKG = 'masters_project' #??? is this right
import rospy


class TestImagePublisher(unittest.TestCase):

    # could test for invalid input

    def __init__(self, *args):

        super(TestImagePublisher, self).__init__(*args)

    def setUp(self):
        rospy.init_node("test_image_publisher")


    def test_wrong_input(self):

