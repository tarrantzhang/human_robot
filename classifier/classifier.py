import sklearn.tree
import pandas as pd
import csv
from sklearn import svm
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
import numpy as np
import random
#data = [[1], [2], [3], [4], [5],[6], [7], [8], [9], [10]];
#target = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
#clf = svm.SVC(gamma=0.001, C=100.);
#clf.fit(data[:-1], target[:-1]);
#print(clf.predict(9));
#print(clf.predict(0));

### code ####
class RandomForest():
	"""docstring for RandomForest"""
	def __init__(self, n_tree, md = 11):
		self.n_tree = n_tree
		self.md = md
		self.tree_bags = []

	def fit(self, X_train, Y_train):
		for i in range(self.n_tree):
			Xb_train,Yb_train = rswr(X_train,Y_train,500);
			tree = DecisionTreeClassifier(max_depth = self.md, max_features = 'auto',min_samples_split=2,)
			tree.fit(X_train, Y_train)
			self.tree_bags.append(tree)

	def predict_proba(self, X_test):
		Y_pred = np.zeros(len(X_test))
		for tree in self.tree_bags:
			Y_pred += tree.predict_proba(X_test)[:,1]
		return 1.0*Y_pred / self.n_tree

def rswr(data,target,num_of_instances):
	#gonna implement random sample with replacement 
    num_list = random.sample(range(1, len(data)), num_of_instances);
    Xb = [];
    Yb = [];
    for num in num_list:
        Xb.append(data[num]);
        Yb.append(target[num]);
    return Xb,Yb

##read data
data = [];
target = [];
with open('features_0-2000_numeric.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    count = 0;
    for row in spamreader:
        if(count != 0):
            target.append(row[0]);
            del row[0];
            for i in range(0,len(row)):
                if(row[i] == ''):
                    row[i] = -999999;
                row[i] = float(row[i]);
            data.append(row);
        count = count+1

test_set = [];
test_target = [];
with open('features_2013_67xx_numeric.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    count = 0;
    for row in spamreader:
        if(count != 0):
            test_target.append(row[0]);
            del row[0];
            for i in range(0,len(row)):
                if(row[i] == ''):
                    row[i] = -999999;
                row[i] = float(row[i]);
            test_set.append(row);
        count = count+1

##simple decision tree
#clf = DecisionTreeClassifier(max_depth = 410,max_features = 'auto',min_samples_split=1,min_samples_leaf =5)
#clf.fit(data[:], target[:]);
#result = clf.predict_proba(test_set)[:,1];

##decision forest
#rf = RandomForest(2,md = 11);
#rf.fit(data[:], target[:])
#result = rf.predict_proba(test_set);

##svm
#clf = svm.SVC(probability = True);
#clf.fit(data[:], target[:]);
#result = clf.predict_proba(test_set)[:,1];

#random forest
rf = RandomForestClassifier(n_estimators = 2,max_depth= 100,min_samples_leaf=1,min_samples_split=2,max_leaf_nodes=160);
rf.fit(data[:], target[:])
result = rf.predict_proba(test_set)[:,1];

##simple decision tree with adaboost
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 410,max_features = 'auto'),n_estimators=50,learning_rate=1.0, algorithm='SAMME.R')
#clf.fit(data[:], target[:]);
#result = clf.predict_proba(test_set)[:,1];

result = result.tolist();
cl = [];
for prb in result:
    if(prb<0.5):
        cl.append("0");
    else:
        cl.append("1");
print(cl);
with open("output.csv", "w",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["prediction"])
    writer.writerows(cl)

result_csv = pd.read_csv("output.csv");
id_csv = pd.read_csv("ids.csv");
output = pd.concat([id_csv, result_csv],axis = 1);
output.to_csv("submission.csv",sep=',',index_label=False,index=False);