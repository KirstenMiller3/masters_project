from cv_bridge import CvBridge, CvBridgeError


# Duplicated method so made as helper
def convert_ROS_to_CV(image, encoding):
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding=encoding)
        return cv_image
    except CvBridgeError, e:
        print e



