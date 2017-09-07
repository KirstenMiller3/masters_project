#!/usr/bin/env python

from sklearn import svm, preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from masters_project.msg import flow_vectors, svm_model, file_input, scaling
from std_msgs.msg import Bool, String
import rospy
import numpy as np
import pickle as p
import traceback

class Machine_learning:
    """
    ROS node that is responsible for fitting the model on the training data. It is also
    used to pickle the training data and the model or load existing training data sets.
    """
    def __init__(self):
        # Sets up node and subscribers
        rospy.init_node('machine_learning', anonymous=True, disable_signals=True)
        rospy.Subscriber("optic_flow_parameters", flow_vectors, self.callback)
        rospy.Subscriber("compute_fit", Bool, self.compute_fit)
        rospy.Subscriber("video_name", String, self.set_training)
        rospy.Subscriber("classification_data", file_input, self.add_data)
        rospy.Subscriber("load_training_data", Bool, self.load_training_data)
        rospy.Subscriber("pickle_training_data", Bool, self.pickle_training_data)

        self.classifications = {}  # where classification data is stored
        self.current_training = None  # Points to the dictionary key of the current training video
        self.model = svm.SVC(C=1.0,  gamma=0.01, kernel='linear', verbose=True, cache_size=1000)
        self.index = 0  # Used for incrementing through classification arrays
        self.X = []  # Array for training data (optic flow vectors)
        self.Y = []  # Array for classifications (1s or 0s)
        self.pub = rospy.Publisher("svm_model", svm_model, queue_size=5)
        # Filenames for output files
        self.x_pickle_filename = 'robot_x.txt'
        self.y_pickle_filename = 'robot_y.txt'
        self.model_filename = 'baxter_optimal.txt'

    # Method to add the optic flow vectors to X and the related classifications to Y
    def callback(self, data):
        x_parameter = list(data.flow_vectors)  # this has list of x and y flow vectors stored as flow_vector objects
        # error checking
        if self.current_training not in self.classifications:
            print "Classifications have not been sent to node"
            exit(0)
        # access the correct classification for that frame
        y_parameter = self.classifications[self.current_training][self.index]
        self.X.append(x_parameter)  # add new training data
        self.Y.extend([y_parameter])  # add new classifications
        self.index += 1  # increment index of classification

    # Once all the training data has been put through the pipeline call this to fit the model
    def compute_fit(self, data):
        # Check to make sure X and Y parameters are correct, if not shut down node
        if self.X == [] or self.Y == [] or len(self.X) is not len(self.Y):
            print "Error: no X or Y data to train model on"
            exit(0)
        # Fit the model to the data
        self.model.fit(self.X, self.Y)
        # Pickle the model and write to file
        s = p.dumps(self.model)
        msg = svm_model()
        msg.pickles = s
        self.pub.publish(msg)
        try:
            model_file = open(self.model_filename, "w")
            model_file.write(s)
            model_file.close()
        except IOError:
            print "Error opening file"

    # Makes sure accessing the correct classifications for the current video
    def set_training(self, data):
        self.current_training = data.data
        self.index = 0  # Reset index

    # Callback method to add the classification data to a dictionary with the filename
    # as the key
    def add_data(self, data):
        self.classifications[data.name] = list(data.classifiers)
        print self.classifications[data.name]

    # callback to pickle and save the training data
    def pickle_training_data(self, data):
        pickle_x = p.dumps(self.X)
        pickle_y = p.dumps(self.Y)
        try:
            x_file = open(self.x_pickle_filename, "w")
            x_file.write(pickle_x)
            x_file.close()

            y_file = open(self.y_pickle_filename, "w")
            y_file.write(pickle_y)
            y_file.close()
            print "Training data Pickled"
        except IOError:
            print "Error opening file"

    # callback to load training data from saved files
    def load_training_data(self, data):
        try:
            x_file = open(self.x_pickle_filename, "r")
            x = x_file.read()
            xp = p.loads(x)
            self.X = xp
            y_file = open(self.y_pickle_filename, "r")
            y = y_file.read()
            self.Y = p.loads(y)
            print "Training data loaded"
        except IOError:
            print "Error opening file"

if __name__ == '__main__':
     try:
        ml = Machine_learning()
        while not rospy.is_shutdown():
            rospy.spin()
     except rospy.ROSInterruptException:
        print 'Shutting down'