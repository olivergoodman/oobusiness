import pandas as pd
import numpy as np
from io import StringIO
df = pd.read_csv('./data/yelp_academic_dataset_business_small.csv')
print(df.head())

print list(df.columns.values)
print('Class labels', np.unique(df['open']))
print df.dtypes

'''
#Partitioning into train/test sets
from sklearn.cross_validation import train_test_split
cols = [col for col in df.columns if col not in ['open']]
X, y = df[cols].values, df['open'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Standardizing data to normal distribution centered at O 
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

#Train logistic regression using L2 regularization
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l2')
lr = LogisticRegression(penalty='l2', C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))
'''
