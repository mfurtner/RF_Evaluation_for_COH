#import necessary libraries
import pandas as pd
import numpy as np
import itertools
import gzip
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

#function that assess variable importance by first dividing collinear
#variables and training many rf models
#train and test df are the best performing train/test folds
#cats are the category lists (including random)
def VarImportance (train_df, test_df, model, cat1, cat2, cat3, cat4):
    #create a list of the four lists
    a = [cat1, cat2, cat3, cat4]
    #create all possible combinations of 1 entry from each list
    #store as lists and create a list of all combination lists
    new_list = [list(ele) for ele in list(itertools.product(*a))]
    #create empty list to store results
    permutation_results = []
    #extract y values from dfs
    y_train = train_df.Target
    y_test = test_df.Target
    #iterate through all lists represnting diff var combos
    for i in range(len(new_list)):
        #set X to only variables in current list
        X_train = train_df[new_list[i]]
        X_test = test_df[new_list[i]]
        #train model
        fit_model = model.fit(X_train, y_train)
        #run permutation importance on fit model
        result = permutation_importance(fit_model, X_test, y_test, random_state=87)
        #convert the permutation importance results in to a pandas 
        #series, sort them, and append them to the empty list
        forest_importances = pd.Series(result.importances_mean, index=X_train.columns)
        forest_importances_sorted = forest_importances.sort_values(ascending=False)
        permutation_results.append(forest_importances_sorted)
        #the result of the above for loop is a list of series, with 
        #the internal lists containing the permutation importance 
        #results for every combination of variables
    #transforming the list of series into a series
    permutation_results_series = pd.Series(permutation_results)
    #transforms the series into a final dataframe, depicting the 
    #permutation importances of the four variables (one from each
    #category and random) in each combination in a separate row
    df_permutation_importance = \
    permutation_results_series.apply(pd.Series)
    #return df
    return df_permutation_importance

#load training folds (list of 10 dfs with labels, targets, variables)
with open('train_folds.pkl', 'rb') as f:
    train_folds = pickle.load(f)

#load test folds (list of 10 dfs with labels, targets, variables)
with open('test_folds.pkl', 'rb') as f:
    test_folds = pickle.load(f)

#extract training and test fold 7 for variable importance
#fold 7 is highest performing fold according to ROC curve
permutation_importance_train = train_folds[6]
permutation_importance_test = test_folds[6]

#set reproducible seeds to add random intergers
rng_train = np.random.default_rng(seed=323)
rng_test = np.random.default_rng(seed=232)

# Add random vars, 0-100 and 0-1000
permutation_importance_train['Random100'] = \
rng_train.integers(0, 101, size=len(permutation_importance_train))
permutation_importance_train['Random1000'] = \
rng_train.integers(0, 1001, size=len(permutation_importance_train))
permutation_importance_test['Random100'] = \
rng_test.integers(0, 101, size=len(permutation_importance_test))
permutation_importance_test['Random1000'] = \
rng_test.integers(0, 1001, size=len(permutation_importance_test))

#create category lists with different types of variables-elevation
#use random intergers between 0-1000 for permutation importance (Cat_4)
Cat_1 = ['Elevation', 'Slope', 'Aspect', 'MeanCurv', 'CasCurv', \
         'GaussCurv', 'ProfCurv', 'TanCurv','TRI', 'VRM', 'VRM7', \
         'Distance_Stream500', 'Distance_Stream500_Elev', \
         'Distance_Stream2k', 'Distance_Stream2k_Elev']
Cat_2 = ['MS_B1', 'MS_B2', 'MS_B3', 'MS_B4', 'CIg', 'EVI', 'GNDVI', \
         'IronOxide', 'MSAVI2', 'MTVI2', 'NDVI', 'NDWI',\
          'SR', 'VARI']
Cat_3 = ['SuperGroup_Transvaal', 'GeoGroup_Chuniespoort', \
         'GeoGroup_Pretoria', 'Formation_Eccles', 'Formation_KarooDolerite',\
         'Formation_Lyttelton', 'Formation_MonteChristo', 'Formation_Oaktree', \
         'Formation_Rooihoogte', 'Formation_TimeballHill', 'Chert_presence', \
         'Diamictite_presence', 'Dolomite_presence', 'Limestone_presence',\
         'Mudrock_presence', 'Quartzite_presence', 'Shale_presence', \
         'Distance_Faults', 'Distance_Dikes']
Cat_4 = ['Random1000']

#set an rf classifier, grows 2000 trees, gini impurity, set random state
rf = RandomForestClassifier(n_estimators=2000, random_state = 99)

rf_permutation_importance_iteration7 = VarImportance(permutation_importance_train, permutation_importance_test, \
                                                     rf, Cat_1, Cat_2, Cat_3, Cat_4)

rf_permutation_importance_iteration7.to_csv('Permutation_Importance_Iteration7.csv')

rf_permutation_importance_iteration7.mean().sort_values(ascending=False)