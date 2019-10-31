import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from joblib import dump
import pandas as pd
from local_utils import FEATURE_DIR, get_cv_iterator, load_data_nozeros_bypoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from dask_ml.model_selection import RandomizedSearchCV, GridSearchCV

def random_forest_grid():
    print("Loading data...", end='\r')
    x, y, iterator = load_data_nozeros_bypoint()
    print("Loaded                      ")
    n_estimators = [int(x) for x in np.linspace(start=50, stop=1000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    scoring = {'accuracy': make_scorer(accuracy_score),
               'prec': make_scorer(precision_score),
               'f1': make_scorer(f1_score),
               'mcc': make_scorer(matthews_corrcoef),
               'rec': make_scorer(recall_score)}

    rf = RandomForestClassifier()
    with ProgressBar():
        grid_search = RandomizedSearchCV(estimator = rf, scoring=scoring, param_distributions=random_grid, refit=False, cv=iterator, n_iter=100, random_state=42, n_jobs = -1)
        grid_search.fit(x, y)

    best_model = grid_search.best_estimator_
    output_path = 'scoring_hypersearch_knn.pickle'
    dump(['test', grid_search.cv_results_], output_path)
    print(grid_search.cv_results_)
    return (grid_search.cv_results_)
    print(best_model)

if __name__ == '__main__':
    rf_grid_results=random_forest_grid()

#write precision to file
    f= open("rf_prec.txt","w+")
    for i in rf_grid_results['split0_test_prec']:
        f.write(str(i))
    for i in rf_grid_results['split1_test_prec']:
        f.write(str(i))
    for i in rf_grid_results['split2_test_prec']:
        f.write(str(i))
    for i in rf_grid_results['split3_test_prec']:
        f.write(str(i))
    for i in rf_grid_results['split4_test_prec']:
        f.write(str(i))

#write f1 to file
    f1= open("knn_f1.txt","w+")
    for i in rf_grid_results['split0_test_f1']:
        f1.write(str(i))
    for i in rf_grid_results['split1_test_f1']:
        f1.write(str(i))
    for i in rf_grid_results['split2_test_f1']:
        f1.write(str(i))
    for i in rf_grid_results['split3_test_f1']:
        f1.write(str(i))
    for i in rf_grid_results['split4_test_f1']:
        f1.write(str(i))


