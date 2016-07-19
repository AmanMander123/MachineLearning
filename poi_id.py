#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import cross_validation
from tester import test_classifier

### Features List
### The first feature must be "poi".
### This features list initialls contains all the features in the 
### dataset (except email_address) 

features_list = ['poi','salary', 'deferral_payments', 
					'total_payments', 'loan_advances', 
					'bonus', 'restricted_stock_deferred', 
					'deferred_income', 'total_stock_value', 
					'expenses', 'exercised_stock_options', 
					'other', 'long_term_incentive', 'restricted_stock', 
					'director_fees','to_messages',  
					'from_poi_to_this_person', 'from_messages', 
					'from_this_person_to_poi', 'shared_receipt_with_poi',
					'bonus_salary_ratio']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#### DATA EXPLORATION ###

### Total number of data points
number_of_data_points = len(data_dict)
#print number_of_data_points

### Allocation across classes
classes = []
for key, value in data_dict.iteritems():
	classes.append(data_dict[key]['poi'])
### Total POI and non-POI observations in the dataset
total_poi = sum(classes)
total_non_poi = len(classes) - total_poi
#print total_poi
#print total_non_poi
### Percent of POIs in the dataset
percent_of_poi = float(total_poi) / float((total_poi + total_non_poi))
#print (percent_of_poi*100)

### Number of features for each observation in the dataset
number_of_features = []
for key, value in data_dict.iteritems():
	number_of_features.append(len(value))
#print (number_of_features)

### Check to see how many NaN's are in each of the features
### Create list of all features
all_features =  data_dict[data_dict.keys()[0]].keys()
### Convert list of features to dictionary and
### initialize values with 0
feature_nan_count = {}
for item in all_features:
	feature_nan_count[item] = 0
### Count number of times an NaN occurs for each feature
for key, value in data_dict.iteritems():
	for item in data_dict[key]:
		if data_dict[key][item] == 'NaN':
			feature_nan_count[item] += 1
#print (feature_nan_count)

### REMOVE OUTLIERS ###

###Create scatterplot of salary and bonus to detect 
### any potential outliers
'''
for key, value in data_dict.iteritems():	
	salary = data_dict[key]['salary']
	bonus = data_dict[key]['bonus']
	plt.scatter(salary, bonus)
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
'''
### The outlier that we see in the above scatterplot is the
### TOTAL line that contains the sum of all the salary and
###  bonus values and needs to be removed
data_dict.pop("TOTAL",0)

### From the PDF file, there is another outlier that needs 
### to be removed
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)

### Create scatterplot of salary and bonus to ensure all outliers
### have been eliminated
'''
for key, value in data_dict.iteritems():	
	salary = data_dict[key]['salary']
	bonus = data_dict[key]['bonus']
	plt.scatter(salary, bonus)
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
'''

### CREATE NEW FEATURE
for key, value in data_dict.iteritems():
	if (data_dict[key]['bonus']) != 'NaN' and \
	(data_dict[key]['salary']) != 'NaN':
		data_dict[key]['bonus_salary_ratio'] = \
		(float(data_dict[key]['bonus'])/float(data_dict[key]['salary']))
	else:
		data_dict[key]['bonus_salary_ratio'] = 'NaN'

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Select K best features
k = 5
skb = SelectKBest(k = k)

### Feature Importance
skb.fit(features, labels)
scores = -np.log10(skb.pvalues_)
scores /= scores.max()
indices = np.argsort(scores)[::-1]
for i in range(6):
    print "feature no. {}: {} ({})".format(i+1,features_list[indices[i]+1],scores[indices[i]])
#print indices[0:k]
updated_features_list = list(features_list[i+1] for i in indices[0:k])

updated_features_list.insert(0,'poi')
#print updated_features_list


### FEATURE SCALING
### Initialize Standard and Min-Max Scaler
Standard_scaler = StandardScaler()
Min_Max_scaler = MinMaxScaler()

### ALGORITHM SELECTION

### Naive Bayes
#clf = Pipeline(steps=[("scaling", Min_Max_scaler),("NaiveBayes",GaussianNB())])
clf = Pipeline(steps=[("NaiveBayes",GaussianNB())])

### Decision Tree
'''
tree = tree.DecisionTreeClassifier()
parameters = {'tree__criterion': ('gini','entropy'),
              'tree__splitter':('best','random'),
              'tree__min_samples_split':[2, 10, 20],
                'tree__max_depth':[10,15,20,25,30],
                'tree__max_leaf_nodes':[5,10,30]}
#pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('pca',PCA(n_components = 2)), ('tree', tree)])
pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('tree', tree)])
#pipeline = Pipeline(steps=[('tree', tree)])

cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='accuracy')
gs.fit(features, labels)
clf = gs.best_estimator_
'''

### VALIDATION

test_classifier(clf, my_dataset, updated_features_list)

### Dump classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, updated_features_list)