import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

from utils.architecture import L2X
from utils.progress import WorkSplitter


class Objective:

    def __init__(self, model_pl, gpu_on, train, valid, test, epoch, seed, save_model, input_dim) -> None:
        """Initialize Class"""
        self.model_pl = model_pl
        self.gpu_on = gpu_on
        self.train = train
        self.valid = valid
        self.test = test
        self.epoch = epoch
        self.seed = seed
        self.save_model = save_model
        self.input_dim = input_dim

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""

        # sample a set of hyperparameters.
        mlp_dim = trial.suggest_categorical('rank', [50, 100, 150, 200])
        selector_dim = trial.suggest_categorical('selector_rank', [50, 100, 150, 200])
        bsize = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024, 2048])
        lr = trial.suggest_categorical('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1])
        l2_reg = trial.suggest_categorical('lambda', [0.001, 0.0001])
        dropout = trial.suggest_categorical('dropout', [0.0])

        #k = trial.suggest_categorical('k', [2, 3, 4, 5, 6, 7])
        temperature = trial.suggest_categorical('temperature', [0.001, 0.01, 0.1, 1])   #[0.001, 0.01, 0.1, 1]

        if self.model_pl.endswith('coat/'):
            k = trial.suggest_categorical('k', [1, 2, 3, 4, 5, 6, 7, 8]) # [1, 2, 3, 4, 5, 6, 7]
        elif self.model_pl.endswith('coat_uniform/'):
            k = trial.suggest_categorical('k', [2, 3, 4, 5, 6, 7, 8])
        elif self.model_pl.endswith('adult/'):
            k = trial.suggest_categorical('k', [2, 3, 4, 5, 6, 7, 8 ,9, 10, 11, 12])
        # else:
        #     k = trial.suggest_categorical('k', [5, 6])
        # if self.model_pl.endswith('syn1/'):
        #     temperature = trial.suggest_categorical('temperature', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        # else:
        #     temperature = trial.suggest_categorical('temperature', [0.001, 0.005, 0.01, 0.05, 0.1])

        setup_seed(self.seed)

        if self.model_pl.endswith('syn1/'):
            activation = 'relu'
        else:
            activation = 'relu'

        model = Trainer(self.epoch, self.model_pl, activation, mlp_dim, selector_dim, self.input_dim, k, temperature,
                        bsize, lr, dropout, l2_reg)
        score = model.train(self.train, self.valid, self.test, self.save_model)

        return score


class Tuner:
    """Class for tuning hyperparameter of MF models."""

    def __init__(self):
        """Initialize Class."""

    def tune(self, n_trials, model_pl, gpu_on, train, valid, test, epoch, seed, save_model, input_dim):
        """Hyperparameter Tuning by TPE."""
        objective = Objective(model_pl=model_pl, gpu_on=gpu_on, train=train, valid=valid, test=test, epoch=epoch,
                              seed=seed, save_model=save_model, input_dim=input_dim)
        study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.trials_dataframe(), study.best_params


class Trainer(object):
    def __init__(self, epoch, model_pt, activation, mlp_dim, selector_dim, input_dim, k, temperature, bsize, lr,
                 dropout, l2_reg):
        self.epoch = epoch
        self.model_pt = model_pt
        self.activation = activation
        self.layer_dims = [mlp_dim, mlp_dim]
        self.bsize = bsize
        self.lr = lr
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.selector_dims = [selector_dim, selector_dim]
        self.input_dim = input_dim
        self.k = k
        self.temperature = temperature

        self.model = L2X(model_pt = self.model_pt, selector_dims=self.selector_dims, layer_dims=self.layer_dims, input_dim=self.input_dim,
                         dropout=self.dropout, activation=self.activation, l2_reg=self.l2_reg, use_bn=True)
        self.criterion = keras.losses.CategoricalCrossentropy(from_logits=False)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.metric = {'test_auc': tf.keras.metrics.AUC(from_logits=False), 'train_loss': tf.keras.metrics.Mean(),
                       'test_loss': tf.keras.metrics.Mean()}

    def train_epoch(self, train_dataset, test_dataset):
        for ds in train_dataset.shuffle(self.bsize, reshuffle_each_iteration=True).batch(self.bsize):
            x, y = ds['x'], ds['y']
            with tf.GradientTape() as tpe:
                y_pred, _ = self.model(x, self.k, self.temperature, is_training=True)
                loss = self.criterion(y, y_pred)
                grads = tpe.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.metric['train_loss'].update_state(loss)

        for ds in test_dataset.batch(self.bsize):
            x, y = ds['x'], ds['y']
            pred, _ = self.model(x, self.k, is_training=False)
            #loss = self.criterion(y, pred)
            #self.metric['test_loss'].update_state(loss)
            self.metric['test_auc'].update_state(y_true=y[:, 1], y_pred=pred[:, 1])

    def train(self, tr_ds, val_ds, te_ds, save_model):
        epoch = self.epoch
        best_epoch, best_metric = 0, 0
        for i in range(epoch):
            self.metric['train_loss'].reset_states()
            self.metric['test_loss'].reset_states()
            self.metric['test_auc'].reset_states()
            self.train_epoch(tr_ds(), val_ds())
            #train_loss = self.metric['train_loss'].result().numpy()
            # print('[Epoch {}]: train loss:{}, test loss:{}, test AUC:{}'.
            #       format(i, train_loss, self.metric['test_loss'].result().numpy(),
            #              self.metric['test_auc'].result().numpy()))

            if self.metric['test_auc'].result().numpy() > best_metric:
                best_metric = self.metric['test_auc'].result().numpy()
                best_epoch = i
                if save_model:
                    tf.saved_model.save(self.model, self.model_pt)

            if i - best_epoch > 10:
                break

        self.metric['test_loss'].reset_states()
        self.metric['test_auc'].reset_states()
        for ds in te_ds().batch(self.bsize):
            x, y = ds['x'], ds['y']
            pred, _ = self.model(x, self.k, is_training=False)
            #loss = self.criterion(y, pred)
            #self.metric['test_loss'].update_state(loss)
            self.metric['test_auc'].update_state(y_true=y[:, 1], y_pred=pred[:, 1])
        test_auc = self.metric['test_auc'].result().numpy()
        print('best_epoch = {}, lr = {}, bsize = {}, dropout = {}, l2_reg = {}, mlp_dims = {}, selector_dim = {}, '
              'k = {}, temperature = {}, test auc = {}'.
              format(best_epoch, self.lr, self.bsize, self.dropout, self.l2_reg, self.layer_dims, self.selector_dims,
                     self.k, self.temperature, test_auc))

        return best_metric


def setup_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def l2x(tr_ds, val_ds, te_ds, model_pl='latents/synthetic_data_syn1/', iteration=300, seed=0, gpu_on=False,
        save_model=False, input_dim=11, n_trials=300, **unused):
    progress = WorkSplitter()

    progress.section("L2X: Set the random seed")
    setup_seed(seed)

    progress.section("L2X: Training")
    tuner = Tuner()
    trials, best_params = tuner.tune(n_trials=n_trials, model_pl=model_pl, gpu_on=gpu_on, train=tr_ds, valid=val_ds,
                                     test=te_ds, epoch=iteration, seed=seed, save_model=save_model,
                                     input_dim=input_dim)
    return trials, best_params
