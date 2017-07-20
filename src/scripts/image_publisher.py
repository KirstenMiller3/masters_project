#!/usr/bin/env python


import rospy
import cv2
import sys
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String


"""
"""
def image_publisher():

    # This node is publishing to the camera_image topic with message type Image.
    pub = rospy.Publisher('camera_image', Image, queue_size=1000)
    # Publisher so machine learning node knows which training matrix to use
    pub2 = rospy.Publisher('video_name', String, queue_size=9)
    # Tells rospy the name of node
    rospy.init_node('image_publisher', anonymous=True) # maybe don't need this as won't normally be more than one node
    bridge = CvBridge()
    rate = rospy.Rate(0.5) # Make publishing rate once every 2 seconds (0.5Hz)

    # If user doesn't enter the video input print error message
    if len(sys.argv) < 2:
        print "You must give an argument to open a video stream."
        print "  It can be a number as video device, e.g.: 0 would be /dev/video0"
        print "  It can be a url of a stream,        e.g.: rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"
        print "  It can be a video file,             e.g.: robotvideo.mkv"
        exit(0)

    resource = sys.argv[1]
    # If we are given just a number, interpret it as a video device
    if len(resource) < 3:
        resource_name = "/dev/video" + resource
        resource = int(resource)
        vidfile = False
    else:
        resource_name = resource
        vidfile = True
    print "Trying to open resource: " + resource_name
    cap = cv2.VideoCapture(resource)
    # Check whether cap is initialized correctly
    if not cap.isOpened():
        print "Error opening resource: " + str(resource)
        print "Maybe opencv VideoCapture can't open it"
        exit(0)

    # Send the name of video to machine_learning node
    if resource == 0:
        resource = "live"
    else:
        resource = resource[:-4]
    pub2.publish(str(resource))

    # Works out the frames per second of the video file/stream
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS) # Get number of frames per second
        print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    else :
        fps = cap.get(cv2.CAP_PROP_FPS) # Get number of frames per second
        print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
    
    # If video is opened correctly start reading video
    print "Correctly opened resource, starting to show feed."
    # Capture frame-by-frame (rval is a bool returned by cap.read() if frame is read correctly)
    rval, frame = cap.read()
    last = None
    loop = 0
    # While video not ended
    while rval:
        # Display the image/frame
        cv2.imshow("Stream: " + resource_name, frame)
        
        # If playing a videofile convert it so is in correct format
        if vidfile and frame is not None:
            frame = np.uint8(frame)

        smaller = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Converts image from cv2 to ros format
        image_message = bridge.cv2_to_imgmsg(smaller, encoding="passthrough")


        if image_message != last and loop % 4 == 0:
            # Publishes frame to camera_image topic
            pub.publish(image_message)

        loop += 1
        last = image_message
        # Get next frame
        rval, frame = cap.read()

        # Introduces a delay of 500 miliseconds so that each frame is published
        key = cv2.waitKey(100)
        
        # exit loop if user presses ESC
        if key == 27 or key == 1048603:
            break
    # Release capture at end and destroy window
    cap.release()
    cv2.destroyWindow("preview")





if __name__ == '__main__':
    try:
        image_publisher()
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass




