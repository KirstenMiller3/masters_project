import pickle as p
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn import svm



class ComputeScore:

    def __init__(self):

        self.prediction_filename = ""
        self.test_filename = ""
        self.trainingX = "baxter_x.txt"
        self.trainingY = "baxter_y.txt"
        self.correct = 0
        self.count = 0
        self.score = 0

    def compute_score(self):

        try:
            with open(self.prediction_filename, "r") as predfile:
                with open(self.test_filename, "r") as testfile:
                    for pline, tline in predfile, testfile:
                        p = pline.split(",")
                        t = tline.split(",")
                        for x,y in p,t:
                            if x == y:
                                self.correct += 1
                            self.count += 1
        except IOError:
            print "The file does not exist"

        self.score = self.correct / self.count * 100
        output = "Total number of frames: "+ self.count +"/n"
        output += "Number of correct classifications: " +self.correct +"/n"
        output+= "Score: " + self.score +"%"
        w = open("prediction_score.txt", "w")
        w.write(output)
        w.close()

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

            output = open("baxter_best_params.txt", "w")
            output.write("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
            output.close()
        except IOError:
            print "The file " + self.trainingX + " does not exist"

if __name__ == '__main__':
    c = ComputeScore()
    c.train_model()
