import argparse

from utils.io import load_yaml
from utils.io import NpyDataset, NpyDataset_public
from utils.modelnames import models
from utils.argcheck import check_int_positive


def main(args):
    params = load_yaml(args.grid)
    params['models'] = {params['models']: models[params['models']]}

    model_path = load_yaml('config/global.yml', key='path')['models']
    save_model = True

    tr_ds = NpyDataset_public(args.path + args.problem + args.train, y_dim=args.y_dim, x_dim=args.x_dim, mode='train')
    val_ds = NpyDataset_public(args.path + args.problem + args.valid, y_dim=args.y_dim, x_dim=args.x_dim, mode='valid')
    te_ds = NpyDataset_public(args.path + args.problem + args.test, y_dim=args.y_dim, x_dim=args.x_dim, mode='test')

    for algorithm in params['models']:
        _, _ = params['models'][algorithm](tr_ds, val_ds, te_ds, model_pl=model_path + args.problem,
                                           iteration=params['iter'], seed=args.seed, gpu_on=args.gpu,
                                           save_model=save_model, input_dim=args.x_dim, n_trials=1)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Pretrain")
    parser.add_argument('-p', dest='path', default="data/")
    parser.add_argument('-d', dest='problem', default="synthetic_data_syn1/")
    parser.add_argument('-t', dest='train', default='train.npy')
    parser.add_argument('-v', dest='valid', default='valid.npy')
    parser.add_argument('-e', dest='test', default='test.npy')
    parser.add_argument('-y', dest='grid', default='config/base.yml')
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=2021)
    parser.add_argument('-xdim', dest='x_dim', type=check_int_positive, default=11)
    parser.add_argument('-ydim', dest='y_dim', type=check_int_positive, default=1)
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    args = parser.parse_args()

    main(args)
