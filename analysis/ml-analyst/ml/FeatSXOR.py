import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import sys
    from feat import Feat
    from sklearn.model_selection import GridSearchCV
    import pdb
    from evaluate_model import evaluate_model
    from read_file import read_file

    dataset = sys.argv[1]
    save_file = sys.argv[2]
    random_seed = int(sys.argv[3])

    print('random_seed:', random_seed)
    # Read the data set into meory
    X,y,names = read_file(dataset,label='label', classification=False )
    # parameter variation
    hyper_params = [{
                    'cross_rate': [0.25,0.5,0.75,1.0],
                    'root_xo_rate': [0.5,0.75,1.0],
                    'fb': [0.0,0.25,0.5,0.75,1.0],
                    'softmax_norm': [True,False]
                   },
                   {
                    'cross_rate': [0.0],
                    'fb': [0.0,0.25,0.5,0.75,1.0],
                    'softmax_norm': [True,False]
                   }]
    # create the classifier
    clf = Feat(obj="fitness,complexity",
               stagewise_xo=True,
               pop_size=500,
               gens=100,
               max_time=600,
               max_stall=10,
               #use_batch=True,
               #batch_size=500,
               ml = "LinearRidgeRegression",
               sel='lexicase',
               surv='nsga2',
               max_depth=6,
               max_dim=min([X.shape[1]*2,50]),
               random_state=random_seed,
               backprop=True,
               iters=10,
               n_threads=1,
               softmax_norm=True,
               verbosity=1)#,
               # logfile=save_file.split('.csv')[0]+'_'+str(random_seed)+'.csv')
  #functions 
    # 10-fold CV score for the pipeline
    clf_name = 'FeatStageXO'
# evaluate the model
    evaluate_model(dataset, save_file, random_seed, clf, clf_name, hyper_params, 
                   classification=False)
