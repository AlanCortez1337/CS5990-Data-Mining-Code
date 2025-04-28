#-------------------------------------------------------------------------
# AUTHOR: Alan Cortez
# FILENAME: knn.py
# SPECIFICATION: 
# FOR: CS 5990- Assignment #4
# TIME SPENT: 
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]
classes += [40] # lets make sure that we have that upper bound also

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
#reading the test data
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
df_train = pd.read_csv('weather_training.csv', sep=',', header=0)
data_training = np.array(df_train.values)
df_train['Temperature (C)'] = pd.cut(df_train['Temperature (C)'], bins=classes, labels=False, right=False)

X_training = np.array(df_train.values)[:, 1:-1]
y_train = np.array(df_train.values)[:, -1:]
y_training = [y_value for sublist in y_train for y_value in sublist]

df_test = pd.read_csv('weather_test.csv', sep=',', header=0)
data_test = np.array(df_test.values)
df_test['Temperature (C)'] = pd.cut(df_train['Temperature (C)'], bins=classes, labels=False, right=False)

X_testing = np.array(df_train.values)[:, 1:-1]
y_test = np.array(df_train.values)[:, -1:]
y_testing = [y_value for sublist in y_test for y_value in sublist]

#loop over the hyperparameter values (k, p, and w) ok KNN
#--> add your Python code here

highest_acc = -1
best_k = -1
best_p = -1
best_w = ''

for k in k_values:
    for p in p_values:
        for w in w_values:

            #fitting the knn to the data
            #--> add your Python code here

            #fitting the knn to the data
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, y_training)

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            #--> add your Python code here

            correct_pred = 0 # TN and TP
            y_pred = clf.predict(X_testing)
            for i in range(len(y_pred)):



                predicted_val = y_testing[i]
                real_val = y_pred[i]

                print("pred:", predicted_val)
                print("real:", real_val)

                if real_val == 0:        
                    prediction_percent = 0         
                else:
                    prediction_percent = abs(100*(abs(predicted_val - real_val)/real_val))


                if prediction_percent <= 15:
                    correct_pred += 1

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            #--> add your Python code here
            accuracy = correct_pred / len(y_pred) # TP + TN / TP + TN + FP + FN

            if accuracy > highest_acc:
                highest_acc = accuracy
                best_k = k
                best_p = p
                best_w = w
                print(f"Highest KNN accuracy so far: {highest_acc}")
                print(f"Parameters: k={k}")
                print(f"Parameters: p={p}")
                print(f"Parameters: w={w}")

print("---------------------")
print("END TESTING")
print("---------------------")
print(f"Highest KNN accuracy: {highest_acc}")
print(f"Parameters: k={best_k}")
print(f"Parameters: p={best_p}")
print(f"Parameters: w={best_w}")