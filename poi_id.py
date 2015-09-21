#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','rescaled_salary',"rescaled_to_poi_fraction","rescaled_from_poi_fraction",'rescaled_shared_receipt_with_poi', 'rescaled_bonus','rescaled_total_stock_value','rescaled_total_payments']
#] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Explore data
print "sample data:",data_dict.values()[0]
print "Number of datapoints:",len(data_dict.values())
numberOfPOI = 0
numberOfNonPOI = 0
for sample in data_dict.values():
    if sample["poi"] == 1:
        numberOfPOI += 1
    elif sample["poi"] == 0:
        numberOfNonPOI += 1
print "Number of POI:",numberOfPOI
print "Number of non-POI",numberOfNonPOI
print "Number of features in dataset:",len(data_dict.values()[0].keys())
from collections import defaultdict
numberOfMissingValuesPerFeature = defaultdict(int)
for sample in data_dict.values():
    for feature in data_dict.values()[0].keys():
        if sample[feature] == "NaN":
            numberOfMissingValuesPerFeature[feature] += 1
import pprint
pp = pprint.PrettyPrinter()
print "Number of missing values per feature:"
pp.pprint(numberOfMissingValuesPerFeature.items())
        
### Task 2: Remove outliers
# import matplotlib.pyplot
# import math
# for (key,sample) in data_dict.items():    
#     salary = sample["salary"]
#     bonus = sample["bonus"]
#     if salary != "NaN" and bonus != "NaN":
#         matplotlib.pyplot.scatter( salary, bonus )
#         if salary > 2.5 * math.pow(10,7) and bonus > 0.8 * math.pow(10,8):
#             print key
# matplotlib.pyplot.xlabel("salary")
# matplotlib.pyplot.ylabel("bonus")
# matplotlib.pyplot.show()
# Looks like the element with key "TOTAL" is an outlier in this dataset, we should remove it
data_dict.pop("TOTAL",0)

### Task 3: Create new feature(s)
maxSalary=float("-inf")
minSalary = float("inf")
maxTotalStockValue =float("-inf")
minTotalStockValue = float("inf")
maxBonus =float("-inf")
minBonus = float("inf")
maxTotalPayments = float("-inf")
minTotalPayments = float("inf")
maxRestrictedStock=float("-inf")
minRestrictedStock = float("inf")
maxToPOIFraction = float("-inf")
minToPOIFraction = float("inf")
maxFromPOIFraction = float("-inf")
minFromPOIFraction = float("inf")
maxSharedReceiptWithPOI = float("-inf")
minSharedReceiptWithPOI = float("inf")

for sample in data_dict.values():
    if sample["from_messages"] != "NaN" and sample["from_messages"] != 0 and sample["from_this_person_to_poi"] != "NaN":
        sample["to_poi_fraction"] = sample["from_this_person_to_poi"] * 1.0/sample["from_messages"]
    else:
        sample["to_poi_fraction"] = 0
    
    if sample["to_poi_fraction"] > maxToPOIFraction:
        maxToPOIFraction = sample["to_poi_fraction"]
    if sample["to_poi_fraction"] < minToPOIFraction:
        minToPOIFraction = sample["to_poi_fraction"]
    
            
        
    if sample["to_messages"] != "NaN" and sample["to_messages"] != 0 and sample["from_poi_to_this_person"] != "NaN":
        sample["from_poi_fraction"] = sample["from_poi_to_this_person"] * 1.0/sample["to_messages"]
    else:
        sample["from_poi_fraction"] = 0
    if sample["from_poi_fraction"] > maxFromPOIFraction:
        maxFromPOIFraction = sample["from_poi_fraction"]
    if sample["from_poi_fraction"] < minFromPOIFraction:
        minFromPOIFraction = sample["from_poi_fraction"]
        
    if sample["shared_receipt_with_poi"] != "NaN":
        if sample["shared_receipt_with_poi"] > maxSharedReceiptWithPOI:
            maxSharedReceiptWithPOI = sample["shared_receipt_with_poi"]
        if sample["shared_receipt_with_poi"] < minSharedReceiptWithPOI:
            minSharedReceiptWithPOI = sample["shared_receipt_with_poi"]
            
    if sample["salary"] != "NaN":
        if sample["salary"] > maxSalary:
            maxSalary = sample["salary"]
        if sample["salary"] < minSalary:
            minSalary = sample["salary"]
    if sample["bonus"] != "NaN":
        if sample["bonus"] > maxBonus:
            maxBonus = sample["bonus"]
        if sample["bonus"] < minBonus:
            minBonus = sample["bonus"]
    if sample["total_stock_value"] != "NaN":
        if sample["total_stock_value"] > maxTotalStockValue:
            maxTotalStockValue = sample["total_stock_value"]
        if sample["total_stock_value"] < minTotalStockValue:
            minTotalStockValue = sample["total_stock_value"]
    if sample["total_payments"] != "NaN":
        if sample["total_payments"] > maxTotalPayments:
            maxTotalPayments = sample["total_payments"]
        if sample["total_payments"] < minTotalPayments:
            minTotalPayments = sample["total_payments"]
    if sample["restricted_stock"] != "NaN":
        if sample["restricted_stock"] > maxRestrictedStock:
            maxRestrictedStock = sample["restricted_stock"]
        if sample["restricted_stock"] < minRestrictedStock:
            minRestrictedStock = sample["restricted_stock"]
print maxSalary, minSalary, maxTotalStockValue, minTotalStockValue, maxBonus, minBonus, maxTotalPayments, minTotalPayments, maxRestrictedStock, minRestrictedStock, maxToPOIFraction, minToPOIFraction, maxFromPOIFraction, minFromPOIFraction
def scaleFeature(maxValue,minValue,value):
    if value != "NaN":
        return ((value - minValue)*1.0)/(maxValue-minValue)
    else:
        return "NaN"
    
for sample in data_dict.values():
    sample["rescaled_salary"] = scaleFeature(maxSalary,minSalary,sample["salary"])
    sample["rescaled_total_stock_value"] = scaleFeature(maxTotalStockValue,minTotalStockValue,sample["total_stock_value"])
    sample["rescaled_bonus"] = scaleFeature(maxBonus,minBonus,sample["bonus"])
    sample["rescaled_to_poi_fraction"] = scaleFeature(maxToPOIFraction,minToPOIFraction,sample["to_poi_fraction"])
    sample["rescaled_from_poi_fraction"] = scaleFeature(maxFromPOIFraction,minFromPOIFraction,sample["from_poi_fraction"])
    sample["rescaled_total_payments"] = scaleFeature(maxTotalPayments,minTotalPayments,sample["total_payments"])
    sample["rescaled_restricted_stock"] = scaleFeature(maxRestrictedStock,minRestrictedStock,sample["restricted_stock"])
    sample["rescaled_shared_receipt_with_poi"] = scaleFeature(maxSharedReceiptWithPOI, minSharedReceiptWithPOI,sample["shared_receipt_with_poi"])
### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Select features
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
kf = KFold(n = len(features),n_folds=10,shuffle=True,random_state=42)
for numOfFeatures in range(1,8):
    ch2 = SelectKBest(k=numOfFeatures)
    features_temp = ch2.fit_transform(features, labels)
    selected_feature_names = [features_list[i+1] for i in ch2.get_support(indices=True)]
    
    
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    
    # Provided to give you a starting point. Try a variety of classifiers.
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
    # Example starting point. Try investigating other evaluation techniques!
    # from sklearn.cross_validation import train_test_split
    # features_train, features_test, labels_train, labels_test = \
    #     train_test_split(features, labels, test_size=0.3, random_state=42)
    
    
    dtF1Scores = []
    dtPrecisionScores = []
    dtRecallScores = []
    print "\n\nPerforming KFold cross validation using Decision Tree Classifier using features:",selected_feature_names,"\n"
    for train_indices,test_indices in kf:
        features_train, features_test = [features_temp[i] for i in train_indices],[features_temp[i] for i in test_indices]
        labels_train, labels_test = [labels[i] for i in train_indices],[labels[i] for i in test_indices]   
        
        dtc = DecisionTreeClassifier(random_state=42)
        from sklearn.grid_search import GridSearchCV
        clf = GridSearchCV(dtc,param_grid = {"min_samples_split":[2,3,4,5,6,7,8,9,10,20,30,40],"max_depth":[2,3,4,5,6,7]})
        clf.fit(features_train, labels_train)
#         print "Decision Tree",clf.best_score_
        print "Decision Tree Parameters:",clf.best_params_
        predictions = clf.predict(features_test)    
        precisionScore = precision_score(y_true=labels_test,y_pred=predictions,pos_label=1)
        recallScore = recall_score(y_true=labels_test,y_pred=predictions,pos_label=1)
        f1Score = f1_score(y_true=labels_test,y_pred=predictions,pos_label=1)
#         print "For Decision Tree Classifier, f1:",f1Score,"precision:",precisionScore,"recallScore:",recallScore
        dtF1Scores.append(f1Score)
        dtPrecisionScores.append(precisionScore)
        dtRecallScores.append(recallScore)
        
    svmF1Scores = []    
    svmPrecisionScores = []
    svmRecallScores = []
    print "\n\nPerforming KFold cross validation using SVC using features:",selected_feature_names,"\n"
    for train_indices,test_indices in kf:
        features_train, features_test = [features_temp[i] for i in train_indices],[features_temp[i] for i in test_indices]
        labels_train, labels_test = [labels[i] for i in train_indices],[labels[i] for i in test_indices]   
        from sklearn.svm import SVC
        svr = SVC(kernel="rbf")
        from sklearn.grid_search import GridSearchCV
        clf = GridSearchCV(svr,param_grid = {"C":[0.1,1,10,100,1000]})
        clf.fit(features_train, labels_train)
#         print "SVC",clf.best_score_
        print "SVC Parameters",clf.best_params_
        predictions = clf.predict(features_test)    
        precisionScore = precision_score(y_true=labels_test,y_pred=predictions,pos_label=1)
        recallScore = recall_score(y_true=labels_test,y_pred=predictions,pos_label=1)
        f1Score = f1_score(y_true=labels_test,y_pred=predictions,pos_label=1)
#         print "For SVM, f1:",f1Score,"precision:",precisionScore,"recallScore:",recallScore 
        svmF1Scores.append(f1Score)
        svmPrecisionScores.append(precisionScore)
        svmRecallScores.append(recallScore)
    print "Average f1 score for DT",sum(dtF1Scores) * 1.0/len(dtF1Scores)
    print "Average precision score for DT",sum(dtPrecisionScores) * 1.0/len(dtPrecisionScores)
    print "Average recall score for DT",sum(dtRecallScores) * 1.0/len(dtRecallScores)
    print "Average f1 score for SVM",sum(svmF1Scores) * 1.0/len(svmF1Scores)
    print "Average precision score for SVM",sum(svmPrecisionScores) * 1.0/len(svmPrecisionScores)
    print "Average recall score for SVM",sum(svmRecallScores) * 1.0/len(svmRecallScores)

#Preparing final classifier based on the observations from the above experiment with
#different values
ch2 = SelectKBest(k=4)
features = ch2.fit_transform(features, labels)
print "feature scores:",ch2.scores_
selected_feature_names = [features_list[i+1] for i in ch2.get_support(indices=True)]
print "selected features for final classifier:",selected_feature_names
features_list = ["poi"]
for feature_name in selected_feature_names:
    features_list.append(feature_name)
clf = DecisionTreeClassifier(min_samples_split=2,max_depth=6,random_state=42)
precisionScores = []
recallScores = []
f1Scores = []
print "Evaluating performance of final classifier that we prepared from above experiments:"
for train_indices,test_indices in kf:
    features_train, features_test = [features[i] for i in train_indices],[features[i] for i in test_indices]
    labels_train, labels_test = [labels[i] for i in train_indices],[labels[i] for i in test_indices]   
    
    dtc = DecisionTreeClassifier()
    clf.fit(features_train,labels_train)
    predictions = clf.predict(features_test)    
    precisionScore = precision_score(y_true=labels_test,y_pred=predictions,pos_label=1)
    recallScore = recall_score(y_true=labels_test,y_pred=predictions,pos_label=1)
    f1Score = f1_score(y_true=labels_test,y_pred=predictions,pos_label=1)
    print "In one of fold within KFolds For Decision Tree Classifier, f1:",f1Score,"precision:",precisionScore,"recallScore:",recallScore
    precisionScores.append(precisionScore)
    recallScores.append(recallScore)
    f1Scores.append(f1Score)
print "Average precision score:",sum(precisionScores)*1.0/len(precisionScores)
print "Average recall score:",sum(recallScores)*1.0/len(recallScores)
print "Average f1 score:",sum(f1Scores)*1.0/len(f1Scores)
clf.fit(features,labels)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)