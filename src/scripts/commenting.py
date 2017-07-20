

def draw_flow(img, flow, step=16):
    # shape returns tuple (rows, columns, channels (if colour image)). This just takes number of
    # rows and stores in h and number of columns and stores it in w
    h, w = img.shape[:2]
    # Create an array with two nested array. The first going from 8 to the height of the image
    # and going up by 16 each time. The second going from 2 to the width of image in steps of 16
    #
    # reshape the array to be a nested array with 2 arrays and the same length as before (guess it
    # just makes sure both the arrays are the same length. Set one to y and x
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)

    fx, fy = flow[y, x].T


    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)


    return vis



# 8 : height of image : 16

# Create an array with two nested array. The first going from 8 to the height of the image
# and going up by 16 each time. The second going from 2 to the width of image in steps of 16

# reshape the array to be a nested array with 2 arrays and the same length as before (guess it
# just makes sure both the arrays are the same length. Set one to y and x

# 7.5 x 10