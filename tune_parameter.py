import argparse
import numpy as np

from utils.io import load_yaml
from utils.io import NpyDataset, NpyDataset_public
from utils.modelnames import models
from utils.argcheck import check_int_positive

from experiment.tuning import hyper_parameter_tuning


from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def main(args):
    params = load_yaml(args.grid)
    parms_models = params['models']
    params['models'] = {params['models']: models[params['models']]}
    
    #ll = [1, 5, 10, 20, 50, 100, 200, 300, 500, 800, 1000]
    #lz = []
    #for i in ll:
    if args.mode:
        print(args.mode)
        print("--------------------------    ",parms_models)
        if parms_models == 'LASSO':
            data = np.load(args.path + args.problem + args.train)
            X, y = data[:, :-1], data[:, -1:]
            estimator = Lasso(alpha=1e-5) #CV(cv=5, normalize=True)
            estimator.fit(X, y)
            print('score : ', np.round(estimator.coef_, 5))
            print("non-zero number : ",np.sum(estimator.coef_ != 0))
            sfm = SelectFromModel(estimator, prefit=True)  # 0.25
            feature_idx = sfm.get_support()
            print(feature_idx)
        elif parms_models == 'TREE':
            data = np.load(args.path + args.problem + args.train)
            X, y = data[:, :-1], data[:, -1:]
            estimator = ExtraTreesClassifier(n_estimators=20)
            estimator.fit(X, y)
            sfm = SelectFromModel(estimator, prefit=True, threshold=0.0001)
            feature_idx = sfm.get_support()
            print(np.sum(feature_idx == True))
            #lz.append(np.sum(feature_idx == True))
        else:
            feature_idx = None
    else:
        feature_idx = None
    #print(lz)
    #exit(0)
    
    print(feature_idx)

    tr_ds = NpyDataset_public(args.path + args.problem + args.train, y_dim=args.y_dim, x_dim=args.x_dim, mode='train')
    val_ds = NpyDataset_public(args.path + args.problem + args.valid, y_dim=args.y_dim, x_dim=args.x_dim, mode='valid')
    te_ds = NpyDataset_public(args.path + args.problem + args.test, y_dim=args.y_dim, x_dim=args.x_dim, mode='test')

    hyper_parameter_tuning(tr_ds, val_ds, te_ds, params, dataset=args.problem,
                           table_save_path=args.problem + args.table_name, seed=args.seed, problem=args.problem,
                           gpu_on=args.gpu, save_model=args.save_model, input_dim=args.x_dim, feature_idx=feature_idx)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")
    parser.add_argument('-tb', dest='table_name', default="l2x_tuning_u.csv")
    parser.add_argument('-p', dest='path', default="data/")
    parser.add_argument('-d', dest='problem', default="synthetic_data_syn1/")
    parser.add_argument('-t', dest='train', default='train.npy')
    parser.add_argument('-v', dest='valid', default='valid.npy')
    parser.add_argument('-e', dest='test', default='test.npy')
    parser.add_argument('-y', dest='grid', default='config/l2x.yml')
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=2021)
    parser.add_argument('-xdim', dest='x_dim', type=check_int_positive, default=11)
    parser.add_argument('-ydim', dest='y_dim', type=check_int_positive, default=1)
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    parser.add_argument('-save', dest='save_model', action='store_true', default=False)
    parser.add_argument('-mode', dest='mode', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
