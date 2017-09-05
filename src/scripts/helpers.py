from cv_bridge import CvBridge, CvBridgeError



def convert_ROS_to_CV(image, encoding):
    try:
        cv_image = CvBridge.imgmsg_to_cv2(image, desired_encoding=encoding)
        return cv_image
    except CvBridgeError, e:
        print e


def convert_CV_to_ROS(image, encoding):
    try:
        cv_image = CvBridge.imgmsg_to_cv2(image, desired_encoding=encoding)
        return cv_image
    except CvBridgeError, e:
        print e



# https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
