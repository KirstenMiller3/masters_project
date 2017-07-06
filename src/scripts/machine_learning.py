#!/usr/bin/env python

# Machine learning node
# Support Vectors are simply the co-ordinates of individual
# observation (e.g. x and y)
# Support vector machines are the lines to split the support vectors
# into different classification groups

'''
from sklearn import svm
# Assumed you have, X (predictor) and Y (target) for training data set
# and x_test(predictor) of test_dataset
# Create SVM classification object
model = svm.svc(kernel='linear', c=1, gamma=1)
# there is various option associated with it, like changing kernel,
# gamma and C value. Will discuss more # about it in next section.Train
# the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)
'''
from masters_project.msg import flow_vectors, flow_vectors_list
import rospy
from std_msgs.msg import String


def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.parameters)
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('machine_learning', anonymous=True)

    rospy.Subscriber("optic_flow_parameters", flow_vectors_list, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()