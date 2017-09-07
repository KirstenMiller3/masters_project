import pickle as p
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn import svm

class ComputeScore:
    """
    This class finds the optimal parameters for a data set to be fit with the model
    """

    def __init__(self):
        self.trainingX = "robot_x.txt"
        self.trainingY = "robot_y.txt"
        self.output = "baxter_best_params.txt"

    # Method for computing optimal parameters for model. Some code is adapted from:
    # http://scikit-learn.org/0.15/auto_examples/grid_search_digits.html
    def train_model(self):

        try:
            with open(self.trainingX, "r") as f:
                m = f.read()
                X = p.loads(m)
                f.close()
            with open(self.trainingY, "r") as g:
                y = g.read()
                Y = p.loads(y)
                g.close()

            C_range = np.logspace(-2, 10, 13)
            smaller_c_range = [0.01,0.1,1.0,10.0,100.0,1000.0]
            gamma_range = np.logspace(-9, 3, 13)
            smaller_gamma_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
            param_grid = dict(gamma=smaller_gamma_range, C=smaller_c_range)
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            print "Searching best parameters"
            grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, n_jobs=3, verbose=4)
            grid.fit(X, Y)
            print("The best parameters are %s with a score of %0.2f"
                % (grid.best_params_, grid.best_score_))
            output = open(self.output, "w")
            output.write("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
            output.close()
        except IOError:
            print "Error opening file"

if __name__ == '__main__':
    c = ComputeScore()
    c.train_model()
