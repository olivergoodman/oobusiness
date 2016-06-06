import pandas as pd
import numpy as np
from io import StringIO
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
print train.dtypes
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
#Partitioning into train/test sets
from sklearn.cross_validation import train_test_split
cols = [col for col in train.columns if col not in ['open']]
X, y = train[cols].values, train['open'].values
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
