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
from masters_project.msg import flow_vectors_list, flow_vectors, svm_model, file_input
from std_msgs.msg import Bool, String
import rospy
import numpy as np
import pickle as p
import time


class Machine_learning:

    def __init__(self):
        rospy.init_node('machine_learning', anonymous=True)

        rospy.Subscriber("optic_flow_parameters", flow_vectors_list, self.callback)
        rospy.Subscriber("compute_fit", Bool, self.compute_fit)
        rospy.Subscriber("video_name", String, self.set_training)
        rospy.Subscriber("classification_data", file_input, self.add_data)
        self.classifications = {}
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
                            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,0,0,0,0]

        self.w = [0,0,1,1,1,1,1,0,0,1,0,0,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]

        self.wavy = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0 ]
        # In this example I set the value of gamma manually. It is possible to automatically find good values for the
        # parameters by using tools such as grid search and cross validation. MAYBE DO THIS?? ASK GERRY
        self.current_training = None
        self.model = svm.SVC(verbose=True)
        # kernel='linear', C=1, gamma=1,
        self.index = 0
        self.X = []
        self.Y = []
        self.pub = rospy.Publisher("svm_model", svm_model, queue_size=5)

    def callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s %s", data.width, data.height)
 
        X = data.parameters  # this has list of x and y flow vectors stored as flow_vector objects
        
        tempX = []
        for i in range(len(X)):
            temp = X[i].flow_vectors
            coords = temp[0].coordinates
            coords2 = temp[1].coordinates

            tempX.append([coords[0], coords[1], coords2[0], coords2[1]])

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
        print len(self.X)
        print len(self.Y)
        print "ENTERED COMPUTE FIT"
        self.model.fit(self.X, self.Y)
        print "X"
        self.model.score(self.X, self.Y)
        print "Y"
        # maybe use cPickle as its 1000 times faster
        s = p.dumps(self.model)
        print "Z"
        msg = svm_model()
        print "A"
        msg.pickles = s
        print "B"
        self.pub.publish(msg)
        print "PUBLISHING"


    def set_training(self, data):
        self.current_training = self.classifications[data.data]
        self.index = 0

    def add_data(self, data):
        self.classifications[data.name] = data.classifiers
        print self.classifications[data.name]

if __name__ == '__main__':
     try:
        ml = Machine_learning()
        rospy.spin()
     except rospy.ROSInterruptException:
        print 'Shutting down'