import sys
import itertools
import pandas as pd
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.metrics import accuracy_score, make_scorer, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline,make_pipeline
from metrics import balanced_accuracy_score
import warnings
import time
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from read_file import read_file
from utils import feature_importance , roc
from convergence import convergence
import pdb
import numpy as np

def mean_corrcoef(x):
    return np.sum(np.square(np.triu(np.corrcoef(x),1)))/(len(x)*(len(x)-1)/2)
    
def benchmark_model(dataset, save_file, random_state, clf, clf_name, hyper_params, classification=True):

    features, labels, feature_names = read_file(dataset,label='target',
                                                classification=classification)

    X_train, X_test, y_train, y_test = train_test_split(features, labels,
							train_size=0.75,
							test_size=0.25,
							random_state=random_state)
    if classification:
        cv = StratifiedKFold(n_splits=5, shuffle=True,random_state=random_state)
    else:
        cv = KFold(n_splits=5, shuffle=True,random_state=random_state)

    if classification:
        scoring = make_scorer(balanced_accuracy_score)
    else:
        scoring = 'r2'

    if len(hyper_params) > 0:
        grid_clf = GridSearchCV(clf,cv=cv, param_grid=hyper_params,
    		            verbose=3,n_jobs=1,scoring=scoring,error_score=0.0)
    else:
        grid_clf = clf
    # print ( pipeline_components)
    # print(pipeline_parameters)
    with warnings.catch_warnings():
        # Squash warning messages. Turn this off when debugging!
        # warnings.simplefilter('ignore')
        print('fitting model\n',50*'=') 
        t0 = time.process_time()
        # generate cross-validated predictions for each data point using the best estimator 
        grid_clf.fit(X_train,y_train)
        
        runtime = time.process_time() - t0
        print('done training. storing results...\n',50*'=')
        if len(hyper_params) > 0:
            best_est = grid_clf.best_estimator_
        else:
            best_est = grid_clf
       
        print('feature correlations...')
        # get the correlation structure of the data transformation 
        phi_cor = 0
        if type(best_est).__name__ == 'Feat' or type(best_est).__name__ == 'MLPRegressorMod':
            phi_cor = mean_corrcoef(best_est.transform(features))
        elif 'RF' not in clf_name:
            phi_cor = mean_corrcoef(features)
        
        print('condition number...')
        # get the condition number of the data transformation
        phi_cn = 0
        if type(best_est).__name__ == 'Feat' or type(best_est).__name__ == 'MLPRegressorMod':
            phi_cn = np.linalg.cond(best_est.transform(features))
        else: 
            phi_cn = np.linalg.cond(features)

        # get the size of the final model
        print('model size...')
        model_size=0
        num_params=0
        # pdb.set_trace()
        if 'Feat' in clf_name:
            # get_dim() here accounts for weights in umbrella ML model
            num_params = best_est.get_n_params()+best_est.get_dim()
            model_size = best_est.get_n_nodes()
        elif 'MLP' in clf_name:
            num_params = np.sum([c.size for c in best_est.coefs_]+
                                [c.size for c in best_est.intercepts_])
        elif 'Torch' in clf_name:
            num_params = best_est.module.get_n_params()
        elif hasattr(best_est,'coef_'):
            num_params = best_est.coef_.size
            model_size = num_params
        elif 'RF' in clf_name:
            model_size = np.sum([e.tree_.node_count for e in best_est.estimators_])
        elif 'XGB' in clf_name:
            model_size = np.sum([m.count(':') for m in best_est._Booster.get_dump()])
        else:
            model_size = features.shape[1]
            num_params = model_size

        # store scores
        try:
            r2_test = r2_score(y_test, best_est.predict(X_test))
        except ValueError as e:
            print(e)
            r2_test = -np.inf
        try:
            mse_test = mean_squared_error(y_test, best_est.predict(X_test))
        except ValueError as e:
            print(e)
            r2_test = -np.inf
        
        param_string = ','.join(['{}={}'.format(p, v) for p,v in best_est.get_params().items()])

        print('saving output...')
        # out.write('dataset\talgorithm\tparameters\tseed\tscore\tcorr\tcond\ttime\tsize\n')
        out_text = '\t'.join([dataset.split('/')[-1].split('.')[0],
                              clf_name,
                              param_string,
                              str(random_state), 
                              str(r2_test),
                              str(mse_test),
                              str(phi_cor),
                              str(phi_cn),
                              str(runtime),
                              str(model_size),
                              str(num_params)])
        print(out_text)
        with open(save_file, 'a') as out:
            out.write(out_text+'\n')
        sys.stdout.flush()
        # evaluate_model(dataset, save_file, random_seed, clf)

        # print('storing cv results')
        # df = pd.DataFrame(data=grid_clf.cv_results_)
        # df['seed'] = random_state
        # cv_save_name = save_file.split('.csv')[0]+'_cv_results.csv'
        # import os.path
        # if os.path.isfile(cv_save_name):
        #     # if exists, append
        #     df.to_csv(cv_save_name, mode='a', header=False, index=False)
        # else:
        #     df.to_csv(cv_save_name, index=False)

        # store convergence logs for xgboost and feat
        # print('storing convergence')
        # convergence(best_est, clf_name, features, labels, save_file, random_state)
