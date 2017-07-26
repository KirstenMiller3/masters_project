#!/usr/bin/env python
import rospy
from masters_project.msg import svm_model, flow_vectors_list
from std_msgs.msg import Bool
import pickle as p
from sklearn import metrics, svm
from sklearn.model_selection import GridSearchCV
import os

class Classifier:

    def __init__(self):
        rospy.init_node('classifier', anonymous=True)

        rospy.Subscriber("svm_model", svm_model, self.callback)
        rospy.Subscriber("time_to_classify", Bool, self.classify)




        self.model = None
        self.X = []
        self.prediction = None
        self.blerg = [0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,0]
        self.w = [0,0,0,1,1,1,1,1,1,0,0,0,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0]
        file = open("prediction_file.txt", "w")
        file = open("prediction_file.txt", "r")
        if not os.stat("prediction_file.txt").st_size == 0:
            m = file.read()
            self.model = p.loads(m)

    def callback(self, data):
        print "I AM ALIVE"
        self.model = p.loads(data.pickles)
        print self.model
        rospy.Subscriber("optic_flow_parameters", flow_vectors_list, self.helper)

    def helper(self, data):
        temp = data.parameters  # this has list of x and y flow vectors stored as flow_vector objects
        print "HELPING"
        for i in range(len(temp)):
            x = temp[i].flow_vectors
            coords = x[0].coordinates
            coords2 = x[1].coordinates

            self.X.append([coords[0], coords[1], coords2[0], coords2[1]])

    def classify(self, data):
        print "CLASSIFY"
        self.prediction = self.model.predict(self.X)
        print self.prediction
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


if __name__ == '__main__':
    try:
        x = Classifier()
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'Shutting down'




