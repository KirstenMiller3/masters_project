#!/usr/bin/env python


import rospy
import cv2
import sys
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String



class Image_Display():
    """
    ROS node that is responsible for accessing an input video stream (via webcam, file, url)
    and displaying it to the user normally, with optic flow visualisation and with HSV optic
    flow visualisation. It is also responsible for converting each frame to a ROS image and
    publishing it.
    """
    def __init__(self):
        # This node is publishing to the camera_image topic with message type Image.
        self.pub = rospy.Publisher('camera_image', Image, queue_size=1000)
        # Publisher so machine learning node knows which training matrix to use
        self.pub2 = rospy.Publisher('video_name', String, queue_size=9)
        rospy.init_node('image_publisher', anonymous=True)

    # Starts the video input and then displays it to the user along with optic flow visualisation and HSV visualisation
    # then does some processing to image and publishes.
    def capture(self):
        # If user doesn't enter the video input print error message and exit
        if len(sys.argv) < 2:
            print "You must give an argument to open a video stream."
            print "  It can be a number as video device, e.g.: 0"
            print "  It can be a url of a stream,        e.g.: rtsp://wow3demo.streamlock.net/vod/mp4:Robo_113k.mov"
            print "  It can be a video file,             e.g.: robotvideo.avi"
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
        if resource == 0:
            resource = "live"
        else:
            resource = resource[:-4]
        # Sleep so subscribers don't miss first message
        rospy.sleep(1)
        # Send the name of video to machine_learning node
        self.pub2.publish(str(resource))
        # If video is opened correctly start reading video
        print "Correctly opened resource, starting to show feed."
        # Capture frame-by-frame
        rval, frame = cap.read()
        prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last = None
        loop = 0
        while rval:
            # Display the image/frame
            cv2.imshow("Stream: " + resource_name, frame)
            cv2.moveWindow("Stream: " + resource_name, 350, 0);
            # Get next frame
            rval, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert frame to grayscale
            # compute optical flow between previous frame and current
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prevgray = gray  # set current frame to prev
            # Display the optic flow visualisation
            flow_image = self.draw_flow(gray, flow)
            cv2.imshow('flow', flow_image)
            cv2.moveWindow("flow", 0, 550);
            # Display the HSV visualisation
            cv2.imwrite(str(self.count) + ".png", flow_image)
            hsv_image = self.draw_hsv(flow)
            cv2.imshow("flow HSV", hsv_image)
            cv2.moveWindow("flow HSV", 700, 550);

            # If playing a videofile convert it so is in correct format
            if vidfile and frame is not None:
                frame = np.uint8(frame)
                smaller = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Converts image from cv2 to ros format
                image_message = self.convert_CV_to_ROS(smaller, "passthrough")
            # Publishes every 4th frame to camera_image topic
            if image_message != last and loop % 4 == 0:
                self.pub.publish(image_message)
            loop += 1
            last = image_message
            key = cv2.waitKey(50)
            # exit loop if user presses ESC
            if key == 27 or key == 1048603:
                break
        # Release capture at end and destroy window
        cap.release()
        cv2.destroyAllWindows()

    # Method to draw the optic flow visualisation taken from:
    # https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
    def draw_flow(self, img, flow, step=16):
        # This method draws the optic flow visualisation between this frame and the last
        # frame
        #
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    # Method to draw the HSV visualisation taken from:
    # https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
    def draw_hsv(self, flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,1] = 255
        hsv[...,2] = np.minimum(v*4, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def convert_CV_to_ROS(self, image, encoding):
        try:
            bridge = CvBridge()
            cv_image = bridge.cv2_to_imgmsg(image, encoding=encoding)
            return cv_image
        except CvBridgeError, e:
            print e


if __name__ == '__main__':
    try:
        x = Image_Display()
        x.capture()
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass




