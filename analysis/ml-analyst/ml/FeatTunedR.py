import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import sys
    from feat import Feat
    from sklearn.model_selection import GridSearchCV
    import pdb
    from benchmark_model import benchmark_model
    from read_file import read_file

    dataset = sys.argv[1]
    save_file = sys.argv[2]
    random_seed = int(sys.argv[3])

    print('random_seed:', random_seed)
    # Read the data set into meory
    X,y,names = read_file(dataset,label='target', classification=False )
    # parameter variation
    hyper_params = {
                    # 'cross_rate': [0.0,0.25,0.5,0.75,1.0],
                    # 'root_xo_rate': [0.5,0.75,1.0],
                    'fb': [0.0,0.25,0.5,0.75,1.0]
                   }
    # create the classifier
    clf = Feat(obj="fitness,complexity",
               pop_size=500,
               gens=200,
               max_time=3600,
               max_stall=50,
               use_batch=True,
               batch_size=1000,
               ml = "LinearRidgeRegression",
               sel='lexicase',
               surv='nsga2',
               max_depth=10,
               max_dim=min([X.shape[1]*2,50]),
               random_state=random_seed,
               backprop=True,
               iters=10,
               n_threads=1,
               verbosity=2,
               # tuned parameters
               cross_rate= 0.75,
               fb = 0.25,
               root_xo_rate = 0.75,
               softmax_norm = False
               # logfile=save_file.split('.csv')[0]+'_'+str(random_seed)+'.log'
               )
  #functions 
    # 10-fold CV score for the pipeline
    clf_name = 'Feat'
# evaluate the model
    benchmark_model(dataset, save_file, random_seed, clf, clf_name, hyper_params, 
                   classification=False)
