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
    # Take an image and returns it with green optic flow lines on it
    # that move as the flow vectors move
    def draw_flow(self, img, flow, step=16):
        # h = number of rows of pixels in img, w = no of columns of pixels
        h, w = img.shape[:2]
        # Create an array with two nested array. The first going from 8 to the height of the image
        # and going up by 16 each time. The second going from 2 cv2.resize
        # to the width of image in steps of 16
        # reshape the array to be a nested array with 2 arrays and the same length as before (guess it
        # just makes sure both the arrays are the same length. Set one to y and x
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        #
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        # Set image to grayscale
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Draw the optic flow lines in green on the grayscale image
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        # draw circles in a grid on the image
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        # return image with optic flow lines drawn on it
        return vis



    def draw_hsv(self, flow):
        # h = number of rows of pixels in img, w = no of columns of pixels
        h, w = flow.shape[:2]
        # fx = w x h view for the red pixels, fy = w x h view for green pixels (bit suspic about this)
        fx, fy = flow[:, :, 0], flow[:, :, 1]

        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx * fx + fy * fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v * 4, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr