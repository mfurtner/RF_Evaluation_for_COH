# RF_Evaluation_for_COH
Python functions and scripts to to evaluate RF models trained on cave and sinkhole sites in the Cradle of Humankind, South Africa, via 10-fold cross validation and permutation importance. 

#### Contains:
train_folds.pkl : a pickle object containing training folds for model cross-validation, readable as a list of 10 dataframes. Training observations include a 'Label' column to designate site or site type, 'Target' column to designate observation type, 'Cluster' column which reflects user-assigned cluster numbers based on the close spatial proximity to other related training observations, and 48 topographic and geomorphological variables derived from remotely sensed imagery of the region. Dataset does not include coordinate information for privacy reasons.

test_folds.pkl : a pickle object containing test folds for model cross-validation, readable as a list of 10 dataframes. Test observations include a 'Label' column to designate site or site type, 'Target' column to designate observation type, 'Cluster' column which reflects user-assigned cluster numbers based on the close spatial proximity to other related training observations, and 48 topographic and geomorphological variables derived from remotely sensed imagery of the region. Dataset does not include coordinate information for privacy reasons.

rf_models folder: contains 10 fit RF models as compressed (gzip) pickle objects
