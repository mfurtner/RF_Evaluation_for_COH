#import necessary libraries
import pandas as pd
import numpy as np
import itertools
import gzip
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import confusion_matrix

#function that will apply folds to dataset and run through rf model, printing scores and saving
#fit models
def fit_fold_rf(train_folds, test_folds, model, drop_list, name_list=None):
    #Create empty lists to hold scores for each iteration
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_list = []
    #For each training iteration
    for i in range(len(train_folds)):
        #Create X and y datasets by dropping columns necessary for X
        #and extracting 'Target' for y
        X_train = train_folds[i].drop(columns=drop_list)
        y_train = train_folds[i].Target
        X_test = test_folds[i].drop(columns=drop_list)
        y_test = test_folds[i].Target
        #Train model
        model.fit(X_train, y_train)
        #Create predictions based on the X test dataset
        y_pred = model.predict(X_test)
        # Create prediction probability for AUC
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        #Calculate evaluation metrics based on the predictions versus the 
        #actual classifications (y test), and add them to their respective lists
        accuracy_list.append(accuracy_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        roc_list.append(roc_auc_score(y_test, y_pred_proba))
        #If name_list is provided, save the trained model using pickle
        #based on the name from name_list
        if name_list:
            model_filename = f"{name_list[i]}.pkl.gz"
            with gzip.open(model_filename, 'wb') as model_file:
                pickle.dump(model, model_file)
    #Create new list to return results
    complete_list = []
    #Add the average of all scores to this list
    complete_list.append(sum(accuracy_list) / len(accuracy_list))
    complete_list.append(sum(precision_list) / len(precision_list))
    complete_list.append(sum(recall_list) / len(recall_list))
    complete_list.append(sum(f1_list) / len(f1_list))
    complete_list.append(sum(roc_list) / len(roc_list))
    #Return the average accuracy, precision, recall, f1 scores, and AUCs (in this order)
    return complete_list

#function that makes average confusion matrix out of all the folds
def make_confusion_matrix (test_folds, model_list, drop_list):
    #create empty lists to hold scores for each iteration
    confusion_list = []
    #for each training iteration:
    for i in range(len(train_folds)):
        #create X and y datasets by dropping columns necessary for X
        #and extracting 'Target' for y
        X_test = test_folds[i].drop(columns=drop_list)
        y_test = test_folds[i].Target
        #load trained model
        model = model_list[i]
        #create predictions based on the X test dataset
        y_pred = model.predict(X_test)
        #calculate confusion matrixes based on the predictions versus
        #the actual classifications (y test)
        #add them to their respective lists
        confusion_list.append(confusion_matrix(y_test, y_pred))
    #creating zero matrix to hold total values
    tot = np.array([[0, 0], [0, 0]])
    #looping through list of matrices and getting total for all
    #positions
    for i in range(len(confusion_list)):
        tot+=confusion_list[i]
    #now divide total array by the length of the list to produce the
    #means
    mean_array = tot/len(confusion_list)
    return mean_array

#load training folds (list of 10 dfs with labels, targets, variables)
with open('train_folds.pkl', 'rb') as f:
    train_folds = pickle.load(f)

#load test folds (list of 10 dfs with labels, targets, variables)
with open('test_folds.pkl', 'rb') as f:
    test_folds = pickle.load(f)

#set an rf classifier, grows 2000 trees, gini impurity, set random state
rf = RandomForestClassifier(n_estimators=2000, random_state = 99)

#drop list for extracting X from dfs
drop_list = ['Label', 'Target', 'Cluster']
#names for saved model iterations
rf_names = ['rf_iteration1', 'rf_iteration2', 'rf_iteration3', 'rf_iteration4', 'rf_iteration5', 'rf_iteration6', 'rf_iteration7', 'rf_iteration8', 'rf_iteration9', 'rf_iteration10']

#fit and train the model 10 times on training/test folds, storing average metrics
rf_results = fit_fold_rf(train_folds, test_folds, rf, drop_list, name_list=rf_names)

#List to store the loaded models
rf_models = []
#Iterate over the model names and load each one, storing them in list
for name in rf_names:
    model_filename = f"{name}.pkl.gz"  #Construct the filename from the model name
    with gzip.open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)  #Load the model
        rf_models.append(model)  #Add the model to the list

#create average cm for 10 model iterations
rf_cm = make_confusion_matrix(test_folds, rf_models, drop_list)

print("rf_results: ", rf_results)
print("cm: ")
print(rf_cm) #displayed [[TN, FP], [FN, TP]]

#create ROC Curves plot
#plot empty figure
plt.figure(figsize=(10, 8))
#Loop over each model and corresponding test_fold
for i, (model, fold) in enumerate(zip(rf_models, test_folds)):
    #Drop the specified columns to extract X
    X_test = fold.drop(columns=drop_list)
    #Extract y
    y_test = fold['Target']
    #Predict probabilities for the positive class
    y_prob = model.predict_proba(X_test)[:, 1]  #Get probabilities for positive class
    #Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    #Interpolate the curve for smoother lines
    mean_fpr = np.linspace(0, 1, 50)
    mean_tpr = np.interp(mean_fpr, fpr, tpr)  #Interpolate to match the new fpr
    #Use model index to create labels
    label = f'Iteration {i+1}'
    #Plot the ROC curve for the current model (smoothed)
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
# Plot random prediction diagonal
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
#create perfect prediction line
x = [0, 0, 1]
y = [0, 1, 1]
plt.plot(x, y, color='black', linestyle=':', linewidth=3)
# Customize plot
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 24, labelpad = 25)
plt.ylabel('True Positive Rate', fontsize = 24, labelpad = 25)
plt.title('ROC Curves for RF Model', fontsize=24, pad = 30)
plt.legend(loc='lower right', fontsize = 14)
plt.tick_params(axis='both', labelsize=18)
plt.grid()
plt.show()