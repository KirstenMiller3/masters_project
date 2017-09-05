#!/usr/bin/env python


import rospy
import sys
from masters_project.msg import file_input


class Classification_Reader():
    """
    ROS node that is responsible for reading data from a CSV text file containing classification
    data. The name of the text file should be the same as the name of the video file the
    classifications refer to.

    This node reads in the information from the file and publishes the classification data
    and the name of the video in a file_input message to the classification_data topic.
    """

    def __init__(self):
        # Set up node and publisher
        rospy.init_node('file_reader', anonymous=True, disable_signals=True)
        pub = rospy.Publisher("classification_data", file_input, queue_size=3)

        # If user doesn't enter a filename print error message and exit
        if len(sys.argv) < 2:
            print "You must give an argument to read a file"
            exit(0)

        # read in the input from the file and store the classification data as an int array
        resource = sys.argv[1]
        msg = file_input()
        count = 0
        try:
            with open(resource, "r") as filestream:
                    for line in filestream:
                        current_line = line.split(",")
                        for i in current_line:
                            msg.classifiers += [int(i)]
                            count += 1
        except IOError:
            print "The file " + resource + " does not exist"
            rospy.signal_shutdown("shutting down")

        # store the name of the file
        msg.name = resource[:-4]
        # sleep to allow subscriber time to join and ensure and message isn't missed
        rospy.sleep(1)
        pub.publish(msg)
        print "Publishing"


if __name__ == '__main__':
    try:
        Classification_Reader()
    except rospy.ROSInterruptException:
        print 'Shutting down'
