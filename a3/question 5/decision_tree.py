# -------------------------------------------------------------------------
# AUTHOR: Alan Cortez
# FILENAME: decision_tree.py
# SPECIFICATION: build three different decision trees and compare the accuracy between them from a test data set
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: This problem probably 1 and a half hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']


def convertData(data_set):
    X = []
    Y = []

    for instance in data_set:
        # Handling X
        new_instance_X = []

        if instance[0] == 'Yes':
            new_instance_X.append(1)
        elif instance[0] == 'No':
            new_instance_X.append(1)
        
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
            Y.append(1)
        else:
            Y.append(2)
    return [X, Y]


for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    X, Y = convertData(data_training)
    
    accuracies = []
    #loop your training and test tasks 10 times here
    for i in range (10):

       #fitting the decision tree to the data by using Gini index and no max_depth
       clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
       clf = clf.fit(X, Y)

       #plotting the decision tree
       tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
       plt.show()

       #read the test data and add this data to data_test NumPy
       #--> add your Python code here
       cheat_test_data = pd.read_csv('cheat_test.csv', sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
       data_test = np.array(cheat_test_data.values)[:,1:] #creating a training matrix without the id (NumPy library)

       test_X, test_Y = convertData(data_test)

       total_correct = 0 # TN + TP
       total_classifications = len(test_Y) # TN + TP + FP + FN

       for index, data in enumerate(test_X):
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            class_predicted = clf.predict([data])[0]
            #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            #--> add your Python code here
            if class_predicted == test_Y[index]:
                total_correct += 1

       #find the average accuracy of this model during the 10 runs (training and test set)
       #--> add your Python code here
       this_accuracy = total_correct / total_classifications
       accuracies.append(this_accuracy)
    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    #--> add your Python code here
    avg_accuracy = np.mean(accuracies)
    print('final accuracy when training on', ds, ':', avg_accuracy)


