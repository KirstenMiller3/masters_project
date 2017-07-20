#!/usr/bin/env python
import rospy
from masters_project.msg import svm_model, flow_vectors_list
from std_msgs.msg import Bool
import pickle as p


class Classifier:

    def __init__(self):
        rospy.init_node('classifier', anonymous=True)

        rospy.Subscriber("svm_model", svm_model, self.callback)
        rospy.Subscriber("time_to_classify", Bool, self.classify)



        self.model = None
        self.X = []

        # spin() simply keeps python from exiting until this node is stopped

    def callback(self, data):
        print "I AM ALIVE"
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
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
        self.model.predict(self.X)



if __name__ == '__main__':
    try:
        x = Classifier()
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'Shutting down'




