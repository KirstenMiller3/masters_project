#!/usr/bin/env python
import rospy
from masters_project.msg import svm_model, flow_vectors
from std_msgs.msg import Bool, String
import pickle as p
import numpy as np
from sklearn import metrics, svm, preprocessing
from sklearn.model_selection import GridSearchCV
import os

class Classifier:

    def __init__(self):
        rospy.init_node('classifier', anonymous=True)

        rospy.Subscriber("svm_model", svm_model, self.callback)
        rospy.Subscriber("time_to_classify", Bool, self.classify)
        rospy.Subscriber("load_existing_model", String, self.load_existing_model)
        rospy.Subscriber("optic_flow_parameters", flow_vectors, self.helper)
        #rospy.Subscriber("to_scale", list, self.scaling)

        self.model = None  # Stores the model for classification
        self.X = []  # Array to store new X values for predictions
        self.prediction = []
        self.scaler = 0
        self.training = []

        f = open("baxter_model.txt", "r")
        m = f.read()
        self.model = p.loads(m)
        print "model loaded"
       # self.scaling()
        #self.scaler = preprocessing.StandardScaler().fit(self.training)

    def scaling(self):
        print "loading..."
        try:
            x_file = open("x_pickle.txt", "r")
            x = x_file.read()
            xp = p.loads(x)
            self.training = xp
            print self.X
        except IOError as e:
            print e

    # Method that is called whenever the node receives and svm_model message from the __ topic
    def callback(self, data):
        print "entered callback"
        try:
            self.model = p.loads(data.pickles)  # unpickle the model
            print self.model
            print "model received and set"
        except p.UnpicklingError:
            print "Not able to unpickle data"

    # Receives the optic flow vectors from the __ topic and adds them to X
    def helper(self, data):
        X = list(data.flow_vectors)
        #x = self.scaler.transform(tempX)
        X = np.reshape(X, (1, -1))
        pred = self.model.predict(X)
        print pred
        self.prediction.append(pred)

        file = open("prediction_file", "a")
        for x in pred:
            file.write(str(x) + ",")
        file.close()
           # self.X.append([coords[0], coords[1], coords2[0], coords2[1]])

    # Maybe make this run the python script if that would avoid threading issues
    def classify(self, data):
        print "CLASSIFY"
        #scaled_x =
        self.prediction = self.model.predict(self.X)
        print self.prediction
        """
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'gamma':
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.2, 0.3, 0.4, 0.5]}
        m = svm.SVC()
        grid = GridSearchCV(estimator=m, param_grid=parameters)

        if len(self.X) < len(self.w):
            self.w = self.w[:len(self.X)]
        else:
            self.X = self.X[:len(self.w)]
        grid.fit(self.X, self.w)
        print(grid)
        print(grid.best_score_)
        print(grid.best_estimator_)
        """
        # getting number of samples error so clearly my if elses aren't working as expected
        # print metrics.accuracy_score(self.prediction, self.w)

        file = open("prediction_file", "w")
        for x in self.prediction:
            file.write(str(x) + ",")
        file.close()

        model_file = open("model_file", "w")
        model = p.dumps(self.model)
        model_file.write(model)
        model_file.close()

    # method so that user can set the model params with custom message
    def set_model_params(self, data):
        self.model = svm.SVC()

    def load_existing_model(self, data):
        try:
            with open(data.data, "r") as f:
                m = f.read()
                self.model = p.loads(m)
        except IOError:
            print "The file " + data.data + " does not exist"

        print self.model
        self.model.set_params(C=1000) # don't think this is relevant anymore
        print "model overwritten"


if __name__ == '__main__':
    try:
        x = Classifier()
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'Shutting down'




