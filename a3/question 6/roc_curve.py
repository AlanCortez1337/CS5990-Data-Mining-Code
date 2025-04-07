# -------------------------------------------------------------------------
# AUTHOR: Alan Cortez
# FILENAME: roc_curve.py
# SPECIFICATION: generate roc curve from cheat_data.csv
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: about 45 minutes for this question
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd
import random

# read the dataset cheat_data.csv and prepare the data_training numpy array
# --> add your Python code here
df = pd.read_csv('cheat_data.csv', sep=',', header=0)
data_training = np.array(df.values)

# # transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# # Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# # be converted to a float.
# # transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]

X = []
y = []

for instance in data_training:
    # Handling X
    new_instance_X = []

    if instance[0] == 'Yes':
        new_instance_X.append(1)
    elif instance[0] == 'No':
        new_instance_X.append(0)
    
    if instance[1] == 'Single':
        new_instance_X.append(1)
        new_instance_X.append(0)
        new_instance_X.append(0)
    elif instance[1] == 'Divorced':
        new_instance_X.append(0)
        new_instance_X.append(1)
        new_instance_X.append(0)
    elif instance[1] == 'Married':
        new_instance_X.append(0)
        new_instance_X.append(0)
        new_instance_X.append(1)

    # '100K' -> ['100', ''] -> '100'
    taxable_income_num = (instance[2].split('k'))[0]
    new_instance_X.append(float(taxable_income_num))

    X.append(new_instance_X)

    # Handling Y
    if instance[3] == 'Yes':
        y.append(1)
    else:
        y.append(0)

# split into train/test sets using 30% for test
# --> add your Python code here
trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.3)

# generate random thresholds for a no-skill prediction (random classifier)
# --> add your Python code here
ns_probs = []
for i in range(len(testX)):
    ns_probs.append(round(random.random(), 2))

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

# keep probabilities for the positive outcome only
# --> add your Python code here
dt_probs = dt_probs[:, 1]

# calculate scores by using both classifiers (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()