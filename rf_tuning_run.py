import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask_ml.model_selection import RandomizedSearchCV, GridSearchCV
import sklearn
from sklearn.ensemble import RandomForestClassifier
import os
from local_utils import FEATURE_DIR, RESULTS_DIR, get_cv_iterator, load_data_nozeros_bypoint
from joblib import dump
import numpy as np
import gouda


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_configurations(results):
    num_configs = len(results['params'])
    means = []
    configs = []
    num_folds = 0
    for x in results.keys():
        if 'test_score' in x and x.startswith('split'):
            num_folds += 1
    print('Num_folds', num_folds)

    for x in range(num_configs):
        m = [results['split%d_test_score' % fold][x] for fold in range(num_folds)]
        print(m)
        means.append(np.mean(m))
        configs.append(results['params'][x])
    return configs, means


# def main_old():
#     y = dd.read_hdf(os.path.join(FEATURE_DIR, 'annotations.hdf5'), 'key')
#     y = y['coverage'] > 0.9
#     x = dd.read_hdf(os.path.join(FEATURE_DIR, 'features.hdf5'), 'key')
#     cols = list(x.columns)
#     cols.remove('fileid')
#     x = x[cols]
#
#     with ProgressBar():
#         x = x.compute().values
#         y = y.compute().values
#
#     params = {
#         'penalty': ['l1', 'l2'],
#         'C': [0.01, .1, 1, 10, 100, 200],
#         'class_weight': [None],  # 'balanced'],
#         'fit_intercept': [True, False]
#     }
#     lr = LogisticRegression()
#     with ProgressBar():
#         grid_search = RandomizedSearchCV(lr,
#                                          params,
#                                          n_jobs=-1,
#                                          cv=get_cv_iterator(),
#                                          refit=True,
#                                          iid=True,
#                                          cache_cv=True)
#         grid_search.fit(x, y)
#     configs, means = get_configurations(grid_search.cv_results_)
#     best_model = grid_search.best_estimator_
#     output_path = 'hypersearch_test_3.pickle'
#     dump(['test', grid_search.cv_results_, best_model], output_path)


def main():
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
    params = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1',
               'mcc': sklearn.metrics.make_scorer(sklearn.metrics.matthews_corrcoef)}

    out_dir = gouda.ensure_dir(os.path.join(RESULTS_DIR, 'log_results'))
    print(out_dir)
    rf = RandomForestClassifier()
    with ProgressBar():
        grid_search = RandomizedSearchCV(rf,
                                         params,
                                         scoring=scoring,
                                         n_jobs=-1,
                                         cv=iterator,
                                         refit='mcc',
                                         iid=True,
                                         cache_cv=True)
        grid_search.fit(x, y)
    configs, means = get_configurations(grid_search.cv_results_)
    output_path = os.path.join(out_dir, 'new_RF.pickle')
    dump(['test', 'grid_search.cv_results_', grid_search.cv_results_,
          'grid_search.best_params_', grid_search.best_params_,
          'grid_search.best_score_', grid_search.best_score_,
          'grid_search.best_estimator_', grid_search.best_estimator_], output_path)



if __name__ == '__main__':
    main()
