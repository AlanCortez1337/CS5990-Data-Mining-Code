#-------------------------------------------------------------------------
# AUTHOR: Alan Cortez
# FILENAME: naive_bayes.py
# SPECIFICATION: Using Gaussian NB lets test various different hyper parameters to see which one gives us the best accuracy
# FOR: CS 5990- Assignment #4
# TIME SPENT: This problem probably took an hour-ish
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]
classes += [40] # lets make sure that we have that upper bound also

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
#--> add your Python code here
df_train = pd.read_csv('weather_training.csv', sep=',', header=0)
data_training = np.array(df_train.values)

#update the training class values according to the discretization (11 values only)
#--> add your Python code here
df_train['Temperature (C)'] = pd.cut(df_train['Temperature (C)'], bins=classes, labels=False, right=False)

X_training = np.array(df_train.values)[:, 1:-1]
y_train = np.array(df_train.values)[:, -1:]
y_training = [y_value for sublist in y_train for y_value in sublist]

#reading the test data
#--> add your Python code here
df_test = pd.read_csv('weather_test.csv', sep=',', header=0)
data_test = np.array(df_test.values)

#update the test class values according to the discretization (11 values only)
#--> add your Python code here
df_test['Temperature (C)'] = pd.cut(df_train['Temperature (C)'], bins=classes, labels=False, right=False)

X_testing = np.array(df_train.values)[:, 1:-1]
y_test = np.array(df_train.values)[:, -1:]
y_testing = [y_value for sublist in y_test for y_value in sublist]

#loop over the hyperparameter value (s)
#--> add your Python code here

highest_acc = -1
best_s = -1

for s in s_values:

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_training, y_training)

    y_pred = clf.predict(X_testing)
    correct_pred = 0 # TN and TP
    for i in range(len(y_pred)):
        #make the naive_bayes prediction for each test sample and start computing its accuracy
        #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
        #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
        #--> add your Python code here
        real_val = y_testing[i]
        pred_val = y_pred[i]

        prediction_percent = abs(100*(abs(pred_val - real_val)/real_val))

        if prediction_percent <= 15:
            correct_pred += 1

        # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
        # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
        # --> add your Python code here
        accuracy = correct_pred / len(y_pred) # TP + TN / TP + TN + FP + FN

        if accuracy > highest_acc:
            highest_acc = accuracy
            best_s = s
            print(f"Highest Naive Bayes accuracy so far: {highest_acc}")
            print(f"Parameters: s={s}")

print("---------------------")
print("END TESTING")
print("---------------------")
print(f"Highest Naive Bayes accuracy: {highest_acc}")
print(f"Parameters: s={s}")



