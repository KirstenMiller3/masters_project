import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



pred_y = open("predicted_y.txt", "r")
x = pred_y.read()
pred = pickle.loads(x)
true_y = open("actual_y.txt", "r")
y = true_y.read()
actual = pickle.loads(y)

class_names = ["waving", "no waving"]



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(actual, pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, class_names, title='Confusion matrix, without normalization')
plt.gcf().subplots_adjust(bottom=0.15)
fig1 = plt.gcf()
fig1.savefig('confusion_matrix.png')

plt.figure()
plot_confusion_matrix(cm=cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix, without normalization')
plt.gcf().subplots_adjust(bottom=0.15)
fig2 = plt.gcf()
fig2.savefig('normalised_confusion_matrix.png')
