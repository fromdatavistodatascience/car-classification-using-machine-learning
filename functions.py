# Functions to be used in the data preparation process

import pandas as pd
import numpy as np
import sklearn.metrics as metric

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree
from IPython.display import Image

from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
classification_report, roc_auc_score)


def market_columns(df):
    """Function that maps multiple entries in a column into individual columns
    and assings a score of 1 or 0 if the entry is question is within a given row.
    """
    categories = []
    for category in list(df['Market Category'].unique()):
        categories += category.split(',')

    unique = set(categories)
    for col in unique:
        df[col] = df['Market Category'].apply(lambda x: 1 if col in x.split(',') else 0)
    df.drop('Market Category', axis=1, inplace=True)
    return df


def onehotencode(X):
    """
    One hot encode the categorical variables in the dataframe to convert them to numerical variables.
    """
    X_obj = X[[col for col,dtype in list(zip(X.columns, X.dtypes)) 
                          if dtype == np.dtype('O')]]
    
    X_nonobj = X[[col for col,dtype in list(zip(X.columns, X.dtypes)) 
                          if dtype != np.dtype('O')]]
    
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_obj_ohe = ohe.fit_transform(X_obj)
    
    X_nonobj_df = pd.DataFrame(X_nonobj).reset_index(drop=True) 
    X_obj_ohe_df = pd.DataFrame(X_obj_ohe.todense(), columns=ohe.get_feature_names()).reset_index(drop=True)
    
    X_all = pd.concat([X_nonobj_df, X_obj_ohe_df], axis=1)
   
    return X_all


def get_metrics_by_class(labels, preds, score_type='macro'):
    
    """
    Return a dataframe of metrics for each class in test data(labels) against the predictions(preds)
    Classes are dataframe index, metrics are columns
    Metrics included are: precision, recall, accuracy and f1_score
    Last row is the the metric for overall model, class_param species the averaging method
    class_param inputs, 'micro', 'macro', 'weighted', 'samples', default is micro
    refer to sklearn.metrics for additional information for more information
    """
    
    metric_dict = {}
    
    for label in set(list(labels)):
        
        #calculate metrics for each class
        pre_score = ((labels==label) & (preds==label)).sum()/((preds==label)).sum()
        rec_score = ((labels==label) & (preds==label)).sum()/((labels==label)).sum()
        
        
        acc_score = (((labels==preds) & (labels==label)).sum()+
                     ((labels!=label) & (preds!=label) & (labels==label)).sum())/(labels==label).sum()
        
        
        f1_score = 2*(pre_score*rec_score)/(pre_score+rec_score)
        
        metric_dict[label] = {'Precision': pre_score,
                              'Recall': rec_score,
                              'Accuracy': acc_score,
                              'F1_Score': f1_score}
        
    metric_dict['M'] = {'Precision': metric.precision_score(labels, preds, average=score_type),
                        'Recall': metric.recall_score(labels, preds, average=score_type),
                        'Accuracy': metric.accuracy_score(labels, preds),
                        'F1_Score': metric.f1_score(labels, preds, average=score_type)}
        
    return pd.DataFrame.from_dict(metric_dict).transpose()

def optimize_knn_params(X, y,  min_k=1, max_k=10, cv=3):
    """
    Using GridSearchCV to find optimal knn parameters for k, weights and metric
    Parameter k and cv are user inputs, while weights and metric are automatically optimized with options below
    Returns the best parameters from grid search
    """
    grid_params = {
        'n_neighbors':list(range(min_k, max_k+1)),
        'weights':['uniform', 'distance'],
        'metric':['minkowski', 'euclidean', 'manhattan']
    }
    
    gs_knn = GridSearchCV(KNeighborsClassifier(), param_grid=grid_params, cv=cv)
    gs_knn.fit(X, y)
    
    return gs_knn.best_params_


# Functions for Decision Trees


def decision_tree(X_train, X_test, y_train, y_test, criterion, max_depth):
    """ Function that takes the train test split results and a decision tree
    criterion as inputs and returns the fitted decision tree and it's
    corresponding confusion metrics."""
    dt = DecisionTreeClassifier(criterion, random_state=10, max_depth=max_depth)
    dt.fit(X_train, y_train)
    y_preds = dt.predict(X_test)
    print('\nConfusion Matrix')
    print('----------------')
    print(pd.crosstab(y_test, y_preds, rownames=['True'],
                      colnames=['Predicted'], margins=True))
    return dt, y_preds, criterion


def plot_feature_importances(X_train, dt):
    """Function that plots a barchart of the individual features and 
    their corresponding feature importance."""
    n_features = X_train.shape[1]
    plt.figure(figsize=(8, 8))
    features = list(zip(X_train.columns, dt.feature_importances_))
    sorted_features = sorted(features, key=lambda x: x[1])   
    sorted_imp = [imp[1] for imp in sorted_features if imp[1] > 0.01]
    sorted_fts = [fts[0] for fts in sorted_features if fts[1] > 0.01]
    plt.barh(range(len(sorted_imp)), sorted_imp, align='center')
    plt.yticks(np.arange(len(sorted_fts)), sorted_fts)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Decision tree features where importance is more than 1%")
    return plt.show()


def tree_image(dt):
    """ Function that takes in a fitted decision tree and returns the 
    corresponding tree image."""
    dot_data = StringIO()
    export_graphviz(dt, out_file=dot_data, filled=True, rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

    return Image(graph.create_png())


def multiclass_roc_auc_score(y, preds, average="macro"):
    """Function to evaluate the AUC ROC score for our multi-class problem."""
    lb = LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)
    preds = lb.transform(preds)
    return roc_auc_score(y, preds, average=average)


# Functions for ADA/Gradient Boosting
def adaboost(train, test, ytrain, ytest):
    """As a performance check, calculate the mean of Adaboost 
    cross validation score for train and test for each model
    """
    adaboost_clf = AdaBoostClassifier()
    adaboost_clf.fit(train, ytrain)
  
    print('Mean Adaboost Cross-Val Score (k=10):')
    cross_val_train = cross_val_score(adaboost_clf, train, ytrain, cv=10, n_jobs=1).mean()
    cross_val_test = cross_val_score(adaboost_clf, test, ytest, cv=10, n_jobs=1).mean()
    return print(f"train: {cross_val_train}, test: {cross_val_test}")

def gbt(train, test, ytrain, ytest):
    """As a performance check, calculate the mean of Gradient Boosting 
    cross validation score for train and test for each model
    """
    gbt_clf = GradientBoostingClassifier()
    gbt_clf.fit(train, ytrain)
    
    print('Mean GBT Cross-Val Score (k=10):')
    cross_val_train = cross_val_score(gbt_clf, train, ytrain, cv=10, n_jobs=1).mean()
    cross_val_test = cross_val_score(gbt_clf, test, ytest, cv=10, n_jobs=1).mean()
    return print(f"train: {cross_val_train}, test: {cross_val_test}")
