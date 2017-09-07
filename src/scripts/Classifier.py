#!/usr/bin/env python
import rospy
from masters_project.msg import svm_model, flow_vectors
from std_msgs.msg import Bool, String
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import re

class Classifier:
    """
    ROS node that is responsible for loading a trained model and then using it to make predictions
    based on optic-flow vector input.
    """
    def __init__(self):
        # Initialise node and set up subscribers
        rospy.init_node('classifier', anonymous=True)
        rospy.Subscriber("svm_model", svm_model, self.callback)
        rospy.Subscriber("load_existing_model", String, self.load_existing_model)
        rospy.Subscriber("optic_flow_parameters", flow_vectors, self.helper)
        rospy.Subscriber("prediction_evaluation", Bool, self.evaluate)
        rospy.Subscriber("pickle_prediction", Bool, self.pickle_prediction)

        self.model = None  # Stores the model for classification
        self.prediction = []  # Where predictions are added
        self.true_y = []  # Correct classifcations

        # Filenames
        self.model_filename = "model.txt"
        self.prediction_filename = "predicted_classifications.txt"
        self.actual_filename = "actual_classifications.txt"

        # Open and load model and classification data of video trying to predict
        try:
            f = open(self.model_filename, "r")
            m = f.read()
            self.model = pickle.loads(m)
            print "model loaded"

            g = open(self.actual_filename, "r")
            for x in g:
                classification = (re.split(',| +', x))
                self.true_y += map(int, classification)
                print "classifications loaded"
        except IOError:
            print "IO error"
            exit(0)


    # Method that is called whenever the node receives a svm_model message
    def callback(self, data):
        try:
            self.model = pickle.loads(data.pickles)
            print "model received and set"
        except pickle.UnpicklingError:
            print "Not able to unpickle data"

    # Receives the optic flow vectors from the optic_flow_parameters topic and makes a prediction
    def helper(self, data):
        X = list(data.flow_vectors)
        X = np.reshape(X, (1, -1))  # Reshape array to be correct input for predict function
        pred = self.model.predict(X)
        print pred
        self.prediction.append(pred)

    # Callback to pickle the predicted and actual classifications
    def pickle_prediction(self, data):
        pickle_x = pickle.dumps(self.prediction)
        pickle_y = pickle.dumps(self.true_y)
        try:
            x_file = open(self.prediction_filename, "w")
            x_file.write(pickle_x)
            x_file.close()

            y_file = open(self.actual_filename, "w")
            y_file.write(pickle_y)
            y_file.close()
            print "Successfully pickled"
        except IOError:
            print "IOError opening files"

    # Give the filename of an existing model to load it
    def load_existing_model(self, data):
        try:
            with open(data.data, "r") as f:
                m = f.read()
                try:
                    self.model = pickle.loads(m)
                    print "model overwritten"
                except pickle.UnpicklingError:
                    print "Not able to unpickle data"
        except IOError:
            print "The file " + data.data + " does not exist"

    # Callback for evaluating the classifications
    def evaluate(self, data):
        score = accuracy_score(self.true_y, self.prediction)
        matrix = confusion_matrix(self.true_y, self.prediction)
        print score
        print matrix


if __name__ == '__main__':
    try:
        x = Classifier()
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'Shutting down'




