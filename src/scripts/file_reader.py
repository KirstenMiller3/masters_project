#!/usr/bin/env python


import rospy
import sys
from masters_project.msg import file_input

def file_reader():
    rospy.init_node('file_reader', anonymous=True)
    pub = rospy.Publisher("classification_data", file_input, queue_size=1)

    # If user doesn't enter a filename print error message
    if len(sys.argv) < 2:
        print "You must give an argument to read a file"
        exit(0)

    resource = sys.argv[1]
    msg = file_input()
    count = 0
    with open(resource, "r") as filestream:
            for line in filestream:
                current_line = line.split(",")
                for i in current_line:
                    msg.classifiers += [int(i)]
                    count += 1
    msg.name = resource[:-4]

    rospy.sleep(1)

    pub.publish(msg)
    print "Publishing"
    #rospy.spin()

if __name__ == '__main__':
    try:
        file_reader()
    except rospy.ROSInterruptException:
        print 'Shutting down'
