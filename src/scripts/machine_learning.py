#!/usr/bin/env python

# Machine learning node
# Support Vectors are simply the co-ordinates of individual
# observation (e.g. x and y)
# Support vector machines are the lines to split the support vectors
# into different classification groups



# INSTEAD OF DOING MACHINE LEARNING THIS NODE IS MORE DOING PICKLING OF VIDEOS AND TRAINING SETS AND THEN
# CALLING .fit() AT THE END!! THEN ANOTHER NODE WILL DO CLASSIFICATIONS
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from masters_project.msg import flow_vectors_list, flow_vectors, svm_model, file_input
from std_msgs.msg import Bool, String
import rospy
import numpy as np
import pickle as p
import traceback



'''
This node enables the software to train to classify things
'''
class Machine_learning:
    # Sets up the instance variables for the class and subscribes to topics and initialises node
    def __init__(self):
        # IS THIS A RIDICULOUS NUMBER OF SUBSCRIBERS?? BAD DESIGN???
        # also don't have topic and method names the same it's confusing
        rospy.init_node('machine_learning', anonymous=True, disable_signals=True)
        rospy.Subscriber("optic_flow_parameters", flow_vectors_list, self.callback)
        rospy.Subscriber("compute_fit", Bool, self.compute_fit)
        rospy.Subscriber("video_name", String, self.set_training)
        rospy.Subscriber("classification_data", file_input, self.add_data)
        rospy.Subscriber("load_existing_model", String, self.load_existing_model)
        # Classifications of frames of different videos
        self.classifications = {}

        # In this example I set the value of gamma manually. It is possible to automatically find good values for the
        # parameters by using tools such as grid search and cross validation. MAYBE DO THIS?? ASK GERRY

        # Variable to point to the dictionary key of the current training video
        self.current_training = None
        # classification model (support vector machine)    maybe use parameter server to set the model
        self.model = svm.SVC(C=1, cache_size=200, gamma=0.01, kernel='linear', max_iter=-1, verbose=True)
        # Counter
        self.index = 0
        # Array for training data (optic flow vectors)
        self.X = []
        # Array for classifiers (1s or 0s)
        self.Y = []
        # set up publisher to publish model to classifier node
        self.pub = rospy.Publisher("svm_model", svm_model, queue_size=5)
        # Filenames for output files
        self.x_pickle_filename = "x_pickle.txt"
        self.y_pickle_filename = "y_pickle.txt"
        self.model_filename = "model.txt"


    # Method to add the optic flow vectors to X and the related classifications to Y
    def callback(self, data):
        # used for testing
        rospy.loginfo(rospy.get_caller_id() + "I heard %s %s", data.width, data.height)
 
        X = data.parameters  # this has list of x and y flow vectors stored as flow_vector objects
        
        tempX = [] # array for X (maybe rename)
        # iterate through each optic_flow vector from the image
        for i in range(len(X)):
            temp = X[i].flow_vectors        # access the vectors in each index of parameters array
            coords = temp[0].coordinates    # get x vector ??
            coords2 = temp[1].coordinates   # get y vector ??

            tempX.append([coords[0], coords[1], coords2[0], coords2[1]]) # Append whiiiit?

        # error checking
        if self.current_training not in self.classifications:
            self.shutdown("Error: Classifications for video have not been passed to node")
        # access the correct classification for that frame
        y = self.classifications[self.current_training][self.index]


        print self.index # testing
        print y # testing
        if y == 1:
            Y = np.ones(len(tempX))
        else:
            Y = np.zeros(len(tempX))

        print Y # testing

        self.X.extend(tempX)    # add new training data
        self.Y.extend(Y)        # add new classifications
        self.index += 1         # increment index of classification

    # Method called when latches to the boolean topic compute_fit
    # Fits model to data and then pickles it and publishes it
    # and writes the model to a file
    def compute_fit(self, data):
        #self.training()
        if self.X is None or self.Y is None or len(self.X) is not len(self.Y):
            self.shutdown("Error: no X or Y data to train model on")

        pickleX = p.dumps(self.X)
        pickleY = p.dumps(self.Y)

        x_file = open(self.x_pickle_filename, "w")
        x_file.write(pickleX)
        x_file.close()

        y_file = open(self.y_pickle_filename, "w")
        y_file.write(pickleY)
        y_file.close()
        print len(self.X)
        print len(self.Y)
        print "ENTERED COMPUTE FIT"

        # are we still doing the fit and score here??
        self.model.fit(self.X, self.Y)
        self.model.score(self.X, self.Y)

        s = p.dumps(self.model)
        msg = svm_model()
        msg.pickles = s
        self.pub.publish(msg)
        print "PUBLISHING"

        model_file = open(self.model_filename, "w")  # should this be hardcoded? maybe instance variable
        model_file.write(s)
        model_file.close()

    # Method called when subscribes to topic that sets the current_training to the correct
    # name and resets the index
    def set_training(self, data):
        self.current_training = data.data
        self.index = 0

    # Method called when file_reader node publishes that stores the classifications
    # for different video frames
    def add_data(self, data):
        print "Entered"
        self.classifications[data.name] = list(data.classifiers)
        print self.classifications[data.name]

    #
    def load_existing_model(self, data):
        try:
            with open(data.data, "r") as f:
                m = f.read()
                self.model = p.loads(m)
        except IOError:
            print "The file " + data.data + " does not exist"
            self.shutdown("Shutting down node")
        except p.UnpicklingError as e:
            print "Not able to unpickle file "+ data.data
            self.shutdown("Shutting down node")
        except Exception as e:
            print(traceback.format_exc(e))
            self.shutdown("Shutting down node")

        print self.model


    # had to get rid of this and do it in python script as the threading in ROS nodes was slowing it down
    def training(self):
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
        grid.fit(self.X, self.Y)

        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

    def shutdown(self, message):
        print message
        rospy.signal_shutdown(message)

if __name__ == '__main__':
     try:
        ml = Machine_learning()
        while not rospy.is_shutdown():
            rospy.spin()
     except rospy.ROSInterruptException:
        print 'Shutting down'