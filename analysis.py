import pandas as pd
import numpy as np
from io import StringIO
import datetime
print "Loading data..."
df = pd.read_csv('./data/yelp_academic_dataset_business_smallest.csv')
print(df.head())

print list(df.columns.values)
print('Class labels', np.unique(df['open']))
print df.dtypes

train = pd.DataFrame(df[['open', 'review_count', 'stars', 'latitude', 'longitude']])
#Add state dummy variables
train = pd.concat([train,pd.get_dummies(df['state'], dummy_na=True)], axis=1)
#Add city dummy variables
train = pd.concat([train,pd.get_dummies(df['city'], dummy_na=True)], axis=1)

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
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

#Train logistic regression using L2 regularization
print "===========Logistic Regression=========="
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l2')
lr = LogisticRegression(penalty='l2', C=0.1)
a = datetime.datetime.now()
lr.fit(X_train_std, y_train)
b = datetime.datetime.now()
print "Logistic Regression Training Time: " + str(b - a)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))

#Train decision tree using gini impurity
print "===========Decision Tree=========="
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
a = datetime.datetime.now()
clf.fit(X_train_std, y_train)
b = datetime.datetime.now()
print "Decision Training Time: " + str(b - a)
print('Training accuracy:', clf.score(X_train_std, y_train))
print('Test accuracy:', clf.score(X_test_std, y_test))

#Train gaussian naive bayes classifier
print "===========Naive Bayes==========="
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
a = datetime.datetime.now()
gnb.fit(X_train_std, y_train)
b = datetime.datetime.now()
print "Naive Bayes Training Time: " + str(b - a)  
print('Training accuracy:', gnb.score(X_train_std, y_train))
print('Test accuracy:', gnb.score(X_test_std, y_test))

#Train neural network classifier
print "===========Neural Network==========="
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
a = datetime.datetime.now()
mlp.fit(X_train_std, y_train)
b = datetime.datetime.now()
print "Neural Network Training Time: " + str(b - a)  
print('Training accuracy:', mlp.score(X_train_std, y_train))
print('Test accuracy:', gnb.score(X_test_std, y_test))

''''
#Sequential Backwards Selection
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
        X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self
        
    def transform(self, X):
        return X[:, self.indices_]
        
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
        
#Use SBS to do feature selection using accuracy as criteria
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)
#Plot the accuracy vs the number of features
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()
#Print selected features
k5 = list(sbs.subsets_[8])
print(train.columns[1:][k5])

#Assessing feature importance using Random Forest
print "==========Random Forest=========="
from sklearn.ensemble import RandomForestClassifier
feat_labels = train.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
    feat_labels[f],
    importances[indices[f]]))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
'''