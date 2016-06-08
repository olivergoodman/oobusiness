import pandas as pd
import numpy as np
from io import StringIO
import datetime
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
print "Loading data..."
df = pd.read_csv('./data/yelp_academic_dataset_business_smallest.csv')

print('Class labels', np.unique(df['open']))
print df.dtypes

train = pd.DataFrame(df[['open', 'review_count', 'stars', 'latitude', 'longitude']])
#Add state dummy variables
train = pd.concat([train,pd.get_dummies(df['state'], dummy_na=True)], axis=1)
#Add city dummy variables
train = pd.concat([train,pd.get_dummies(df['city'], dummy_na=True)], axis=1)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve

#Learning Curve Plotting Function
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

'''
#Adding distances from center of city
from geopy import geocoders
gn = geocoders.GeoNames(username="tangdrew")
distance = []
unique_cities = list(set(df['city'] + ", " + df['state']))
print unique_cities

for i in range(0,unique_cities):
    print unique_cities[i]
    lat = gn.geocode(unique_cities[i]).latitude
    long = gn.geocode(unique_cities[i]).longitude
    distance.append(np.sqrt((df['latitude'][i] - lat)**2 + (df['longitude'][i] - long)**2))
train['d2d'] = distance

'''
print "Creating training and validation sets..."
#Partitioning into train/test sets
from sklearn.cross_validation import train_test_split
cols = [col for col in train.columns if col not in ['open']]
X, y = train[cols].values, train['open'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print "Normalizing data..."
#Standardizing data to normal distribution centered at O 
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

#CV for learning curve
cv = cross_validation.ShuffleSplit(X_std.shape[0], n_iter=10, test_size=0.3, random_state=0)

#Train logistic regression using L2 regularization
print "===========Logistic Regression=========="
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l2')
lr = LogisticRegression(penalty='l2', C=0.1)
a = datetime.datetime.now()
lr.fit(X_train_std, y_train)
b = datetime.datetime.now()
y_pred_lr = lr.predict(X_test_std)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
print "Logistic Regression Training Time: " + str(b - a)
print "Training accuracy: ", lr.score(X_train_std, y_train)
print "Test accuracy: ", lr.score(X_test_std, y_test)
print "Test Precision: ", precision
print "Test Recall: ", recall

#Train decision tree using gini impurity
print "===========Decision Tree=========="
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dt = DecisionTreeClassifier(random_state=0)
a = datetime.datetime.now()
dt.fit(X_train_std, y_train)
b = datetime.datetime.now()
y_pred_dt = dt.predict(X_test_std)
precision = precision_score(y_test, y_pred_dt)
recall = recall_score(y_test, y_pred_dt)
print "Decision Tree Training Time: " + str(b - a)
print "Training accuracy: ", dt.score(X_train_std, y_train)
print "Test accuracy: ", dt.score(X_test_std, y_test)
print "Test Precision: ", precision
print "Test Recall: ", recall
#Export tree visualization
print "Exporting tree visualization..."
#tree.export_graphviz(dt,out_file='tree.dot')
#Plotting Learning Curve
print "Plotting learning curve..."
title = "Learning Curves (Logistic Regression)"
plot_learning_curve(dt, title, X, y, ylim=(0.0, 1.01), cv=cv)
plt.show()

#Train gaussian naive bayes classifier
print "===========Naive Bayes==========="
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
a = datetime.datetime.now()
gnb.fit(X_train_std, y_train)
b = datetime.datetime.now()
y_pred_gnb = gnb.predict(X_test_std)
precision = precision_score(y_test, y_pred_gnb)
recall = recall_score(y_test, y_pred_gnb)
print "Naive Bayes Training Time: " + str(b - a)
print "Training accuracy: ", gnb.score(X_train_std, y_train)
print "Test accuracy: ", gnb.score(X_test_std, y_test)
print "Test Precision: ", precision
print "Test Recall: ", recall
'''
#Train support vector machine
print "===========Support Vector Machine==========="
from sklearn.svm import SVC
svm = SVC()
a = datetime.datetime.now()
svm.fit(X_train_std, y_train)
b = datetime.datetime.now()
y_pred_svm = svm.predict(X_test_std)
precision = precision_score(y_test, y_pred_svm)
recall = recall_score(y_test, y_pred_svm)
print "Support Vector Machine Training Time: " + str(b - a)
print "Training accuracy: ", svm.score(X_train_std, y_train)
print "Test accuracy: ", svm.score(X_test_std, y_test)
print "Test Precision: ", precision
print "Test Recall: ", recall
'''
print "Writing predictions to csv..."
np.savetxt("pred_lr.csv", y_pred_lr, delimiter=",")
np.savetxt("pred_dt.csv", y_pred_dt, delimiter=",")
np.savetxt("pred_gnb.csv", y_pred_gnb, delimiter=",")
#np.savetxt("pred_lr.csv", y_pred_lr, delimiter=",")
