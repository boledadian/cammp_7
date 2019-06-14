import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

<<<<<<< HEAD
=======
df = pd.read_excel('Cr-poisoning.xlsx')
X = df.iloc[:, 0:2]
y = df.iloc[:, 2]

le = LabelEncoder()
y = le.fit_transform(y.values)

scaler = StandardScaler()
X = scaler.fit_transform(X)
# y = scaler.fit(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.75)

clf = SVC(kernel='linear', gamma=0.5, C=20)
# knn = KNeighborsClassifier(n_neighbors=4)
# rf = RandomForestClassifier()
log_reg = LogisticRegression()
model = clf
model.fit(X_train, y_train)
#
y_hat = model.predict(X_test)

confusion_mat = confusion_matrix(y_test, y_hat)
print(confusion_mat)
# print(classification_report(y_test, y_hat))
print(accuracy_score(y_test, y_hat))
>>>>>>> andrei

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, model, xx, yy, **params):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


df = pd.read_excel('Cr-poisoning.xlsx')
X = df.iloc[:, 0:2]
y = df.iloc[:, 2]

le = LabelEncoder()
y = le.fit_transform(y.values)

scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.75)

clf = SVC(kernel='linear', gamma=0.5, C=20)
knn = KNeighborsClassifier(n_neighbors=4)
rf = RandomForestClassifier()
log_reg = LogisticRegression()


models = [clf,knn,rf,log_reg]
modelNames = ["SVM", "KNN", "RandomForest", "LogisticRegression"]

for i, model in enumerate(models):
    print(modelNames[i])
    if model == nn:
        model.fit(X_train,y_train,epochs=150,batch_size = 4)
    else:
        model.fit(X_train, y_train)
    
    y_hat = model.predict(X_test)

    confusion_mat = confusion_matrix(y_test, y_hat)
    print(confusion_mat)

    print(accuracy_score(y_test, y_hat))




    print_confusion_matrix(confusion_mat, ["1", "2", "3", "4", "5"])
    plt.savefig("confusion_matrix_" + modelNames[i]  + ".png")

    fig, ax = plt.subplots()
    title = 'Decision Boundary for ' + modelNames[i]
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # plot_contours(ax, clf, xx, yy, cmap=plt.cm.get_cmap('Dark2'), alpha=0.8)
    plot_contours(ax, model, xx, yy, cmap=plt.cm.get_cmap('Dark2'), alpha=0.8)
    #ax.scatter(X0, X1, c=y, s=20, cmap=plt.cm.get_cmap('Dark2'), edgecolors='k')
    #plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    # Plot also the training points
    col_map = {'SrO': 'olive', 'SrCrO3': 'yellow', 'Sr3Cr2O8': 'magenta','Sr2CrO4':"blue",'SrCrO4':"navy"}
    plt.scatter(X0, X1, c= [col_map[lb] for lb in df['reaction product']], edgecolors='k', cmap=plt.cm.Paired)
    ax.set_ylabel('$log_{10}pCrO3_3$')
    ax.set_xlabel('$log_{10}pO_2$')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    # ax.legend()
    plt.savefig('classification_' + modelNames[i] + '.png')
    plt.show()
