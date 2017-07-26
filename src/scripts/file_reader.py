#!/usr/bin/env python


import rospy
import sys
from masters_project.msg import file_input


def file_reader():
    rospy.init_node('file_reader', anonymous=True)
    pub = rospy.Publisher("classification_data", file_input, queue_size=3)

    # If user doesn't enter a filename print error message
    if len(sys.argv) < 2:
        print "You must give an argument to read a file"
        exit(0)


    # read in the input from the file and store it as a file_input message
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

    msg.name = resource[:-4]

    # sleep to allow subscribers and message isn't missed
    rospy.sleep(1)

    pub.publish(msg)
    print "Publishing" # testing


if __name__ == '__main__':
    try:
        file_reader()
    except rospy.ROSInterruptException:
        print 'Shutting down'
