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

# INSTEAD OF DOING MACHINE LEARNING THIS NODE IS MORE DOING PICKLING OF VIDEOS AND TRAINING SETS AND THEN
# CALLING .fit() AT THE END!! THEN ANOTHER NODE WILL DO CLASSIFICATIONS
from sklearn import svm
from masters_project.msg import flow_vectors_list, flow_vectors
from std_msgs.msg import Bool, String
import rospy
import numpy as np
import pickle


class Machine_learning():

    def __init__(self):
        self.live = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0]

        self.wave = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0]

        self.w = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0]

        self.wavy = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0 ]
        # In this example I set the value of gamma manually. It is possible to automatically find good values for the
        # parameters by using tools such as grid search and cross validation. MAYBE DO THIS?? ASK GERRY
        self.current_training = None
        self.model = svm.SVC(kernel='linear', C=1, gamma=1)
        self.index = 0
        self.X = []
        self.Y = []

    def callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s %s", data.width, data.height)
 
        X = data.parameters # this has list of x and y flow vectors stored as flow_vector objects
        
        tempX = []
        for i in range(len(X)):
            temp = X[i].flow_vectors
            tempX.append([temp[0], temp[1]])

        y = self.current_training[self.index]
        print self.index
        print y 
        if y == 1:
            Y = np.ones(len(tempX))
        else:
            Y = np.zeros(len(tempX))

        print Y
        #self.model.fit(tempX, Y)
        #self.model.score(X, self.training[self.frame])
        self.X.extend(tempX)
        self.Y.extend(Y)
        self.index += 1


    def compute_fit(self, data):
        #print self.X
        #print self.Y
        self.model.fit(self.X, self.Y)
        print self.model.score(self.X, self.Y)


    def set_training(self, data):
        self.current_training = getattr(self, data.data)
        self.index = 0


    def listener(self):
        # In ROS, nodes are uniquely named. If two nodes with the same
        # node are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('machine_learning', anonymous=True)

        rospy.Subscriber("optic_flow_parameters", flow_vectors_list, self.callback)
        rospy.Subscriber("compute_fit", Bool, self.compute_fit)
        rospy.Subscriber("video_name", String, self.set_training)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    # Helper function to convert the subscribed data into an image shape
    def convert_parameters(self, w, h, f):
        flow_matrix = [[0 for x in range(w)] for y in range(h)]

        count = 0
        for i in range(h - 1):
            for j in range(w - 1):
                flow_matrix[i][j] = f[count]
                count += 1

        return flow_matrix

    def svm(self):
        self.model.fit(self.X, self.training)
        print self.model.score(self.X, self.training)


if __name__ == '__main__':
    try:
        ml = Machine_learning()
        ml.listener()
    except rospy.ROSInterruptException:
        print 'Shutting down'


