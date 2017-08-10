from sklearn import svm
import pickle
from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

clf = svm.SVC(C=100.0,  gamma=0.001, kernel='linear', verbose=True, cache_size=1000)
clf.fit(digits.data[:-1], digits.target[:-1])
print clf.predict(digits.data[-1:])





"""
try:
    x_file = open("full_x_pickle.txt", "r")
    x = x_file.read()
    xp = pickle.loads(x)
    X = xp
    print X
    y_file = open("full_y_pickle.txt", "r")
    y = y_file.read()
    Y = pickle.loads(y)
    print Y
    test_file = open("testX.txt", "r")
    t = test_file.read()
    test = pickle.loads(t)
except IOError as e:
    print e

clf.fit(X,Y)

p = clf.predict(test)

f = open("sklearn.txt", "a")
for x in p:
    f.write(str(x) + ",")
f.close()
"""