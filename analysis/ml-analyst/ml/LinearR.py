
import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import sys
    import numpy as np

    from sklearn.linear_model import ElasticNet 
    from benchmark_model import benchmark_model

    # inputs
    dataset = sys.argv[1]
    save_file = sys.argv[2]
    random_seed = int(sys.argv[3])

    print('random_seed:', random_seed)

    np.random.seed(random_seed)

    # parameters for method
    hyper_params = {
            'l1_ratio': np.linspace(0,1,100),
            'selection': ['cyclic','random']
            }
     
    clf = ElasticNet()
    clf_name = 'ElasticNet' 
    #evaluate
    benchmark_model(dataset, save_file, random_seed, clf, clf_name, hyper_params,
                   classification=False)
