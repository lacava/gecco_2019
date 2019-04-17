import multiprocessing


 
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    import sys
    import numpy as np

    from MLPRegressorMod import MLPRegressorMod
    from evaluate_model import evaluate_model
    # inputs
    dataset = sys.argv[1]
    save_file = sys.argv[2]
    random_seed = int(sys.argv[3])

    print('random_seed:', random_seed)

    np.random.seed(random_seed)

    # parameters for method
    hyper_params = {
            'hidden_layer_sizes': [(100,),(100,50,10),(100,50,20,10,10,8)],
            'activation': ['logistic','tanh','relu'],
            'solver': ['lbfgs','adam'],
            'learning_rate_init': np.logspace(-4,-2,3),
            'alpha':np.logspace(-5,-3,3)
            }

    clf = MLPRegressorMod(early_stopping=True, 
                       max_iter=10000,
                       random_state=random_seed)
    clf_name = 'MLP' 
    #evaluate
    evaluate_model(dataset, save_file, random_seed, clf, clf_name, hyper_params,
                   classification=False)
