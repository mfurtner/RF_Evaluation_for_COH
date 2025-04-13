# RF_Evaluation_for_COH
Python functions and scripts to to evaluate RF models trained on cave and sinkhole sites in the Cradle of Humankind (CoH), South Africa, via 10-fold cross validation and permutation importance. 

#### Contains:
train_folds.pkl : a pickle object containing training folds for model cross-validation, readable as a list of 10 dataframes. Training observations include a 'Label' column to designate site or site type, 'Target' column to designate observation type, 'Cluster' column which reflects user-assigned cluster numbers based on the close spatial proximity to other related training observations, and 48 topographic and geomorphological variables derived from remotely sensed imagery of the region. Dataset does not include coordinate information for privacy reasons.

test_folds.pkl : a pickle object containing test folds for model cross-validation, readable as a list of 10 dataframes. Test observations include a 'Label' column to designate site or site type, 'Target' column to designate observation type, 'Cluster' column which reflects user-assigned cluster numbers based on the close spatial proximity to other related training observations, and 48 topographic and geomorphological variables derived from remotely sensed imagery of the region. Dataset does not include coordinate information for privacy reasons.

rf_cross_validation.py : Python functions and scripts to perform 10-fold cross valiation on RF models trained on topographic and geomporphological variables of cave/sinkhole sites in the CoH, producing average metrics including accuracy, precision, recall, F1 score, AUC, as well as an average confusion matrix and ROC curves for each iteration.

rf_permutation_importance.py : Python functions and scripts to perform permutation importance assessments on individual variables for the RF model using the highest performing training/test folds. A specific function was written to keep collinear variables separate to better assess their indiviudal influences on model success.

rf_models folder: contains 10 fit RF models as compressed (gzip) pickle objects, an output of rf_cross_validation.py, to be used in subsequent analyses
