#!/usr/bin/env python
import rospy
from masters_project.msg import svm_model, flow_vectors_list
from std_msgs.msg import Bool, String
import pickle as p
from sklearn import metrics, svm
from sklearn.model_selection import GridSearchCV
import os

class Classifier:

    def __init__(self):
        rospy.init_node('classifier', anonymous=True)

        rospy.Subscriber("svm_model", svm_model, self.callback)
        rospy.Subscriber("time_to_classify", Bool, self.classify)
        rospy.Subscriber("load_existing_model", String, self.load_existing_model)
        rospy.Subscriber("optic_flow_parameters", flow_vectors_list, self.helper)

        self.model = None  # Stores the model for classification
        self.X = []  # Array to store new X values for predictions
        self.prediction = []
        self.scaler = 0

        f = open("gerry_model.txt", "r")
        m = f.read()
        self.model = p.loads(m)
        print "model loaded"


    # Method that is called whenever the node receives and svm_model message from the __ topic
    def callback(self, data):
        print "entered callback"
        try:
            self.model = p.loads(data.pickles)  # unpickle the model
            self.scaler = data.scaler
            print self.model
            print "model received and set"
        except p.UnpicklingError:
            print "Not able to unpickle data"

    # Receives the optic flow vectors from the __ topic and adds them to X
    def helper(self, data):
        temp = data.parameters  # this has list of x and y flow vectors stored as flow_vector objects
        print "HELPING"
        for i in range(len(temp)):
            x = temp[i].flow_vectors
            coords = x[0].coordinates
            coords2 = x[1].coordinates
            test = [[coords[0], coords[1], coords2[0], coords2[1]]]
            prediction = self.model.predict(test)
            print prediction
            self.prediction.append(prediction)

            file = open("prediction_file", "a")
            for x in prediction:
                file.write(str(x) + ",")
            file.close()
           # self.X.append([coords[0], coords[1], coords2[0], coords2[1]])

    # Maybe make this run the python script if that would avoid threading issues
    def classify(self, data):
        print "CLASSIFY"
        scaled_x =
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




