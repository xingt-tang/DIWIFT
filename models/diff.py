import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

from utils.io import SubNpyDataset
from utils.architecture import DIFF1, DIFF2, DIFF_no_att
from utils.progress import WorkSplitter


class Objective:

    def __init__(self, our_model_pl, model_pl, gpu_on, train, valid, test, epoch, seed, save_model, input_dim) -> None:
        """Initialize Class"""
        self.our_model_pl = our_model_pl
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


        diff_bsize = trial.suggest_categorical('diff_batch_size', [7000]) #7000 15000

        retrain_mlp_dim = trial.suggest_categorical('retrain_rank', [50]) # 50 200 

        # sample a set of hyperparameters.
        diff_mlp_dim = trial.suggest_categorical('rank', [50])
        selector_dim = trial.suggest_categorical('selector_rank', [50, 100, 150, 200])
        retrain_bsize = trial.suggest_categorical('retrain_batch_size', [512, 1024, 2048, 7000]) #7000 10000
        lr = trial.suggest_categorical('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1])
        l2_reg = trial.suggest_categorical('lambda', [0.001])
        dropout = trial.suggest_categorical('dropout', [0.0])
        temperature = trial.suggest_categorical('temperature', [1])
        heads = trial.suggest_categorical('heads', [1])
        tbsize = trial.suggest_categorical('test_batch_size', [7000])  
        hvp_len = trial.suggest_categorical('hvp_len', [1])
        hvp_iter = trial.suggest_categorical('hvp_iter', [1000])
        hvp_size = trial.suggest_categorical('hvp_size', [100])
        hvp_scale = trial.suggest_categorical('hvp_scale', [500])


        setup_seed(self.seed)

        if self.model_pl.endswith('syn1/'):
            activation = 'relu'
        else:
            activation = 'relu'

        model = Trainer(self.epoch, self.model_pl, activation, diff_mlp_dim, selector_dim, self.input_dim, temperature,
                        diff_bsize, lr, dropout, l2_reg, heads, tbsize, hvp_len, hvp_iter, hvp_size, hvp_scale)
        influence_loss = model.train(self.train, self.valid, self.our_model_pl)

        model = ReTrainer(self.epoch, self.our_model_pl, activation, retrain_mlp_dim, retrain_bsize, lr, dropout, l2_reg)
        score = model.train(self.train, self.valid, self.test, self.save_model, influence_loss)

        return score


class Tuner:
    """Class for tuning hyperparameter of MF models."""

    def __init__(self):
        """Initialize Class."""

    def tune(self, n_trials, our_model_pl, model_pl, gpu_on, train, valid, test, epoch, seed, save_model, input_dim):
        """Hyperparameter Tuning by TPE."""
        objective = Objective(our_model_pl=our_model_pl, model_pl=model_pl, gpu_on=gpu_on, train=train, valid=valid,
                              test=test, epoch=epoch, seed=seed, save_model=save_model, input_dim=input_dim)
        study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.trials_dataframe(), study.best_params


class Trainer(object):
    def __init__(self, epoch, model_pt, activation, mlp_dim, selector_dim, input_dim, temperature, bsize, lr, dropout,
                 l2_reg, heads, tbsize, hvp_len, hvp_iter, hvp_size, hvp_scale):
        self.epoch = epoch
        self.model_pt = model_pt
        self.activation = activation
        self.layer_dims = [mlp_dim, mlp_dim]
        self.bsize = bsize
        self.lr = lr
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.heads = heads
        self.selector_dims = [selector_dim, selector_dim]
        self.input_dim = input_dim
        self.temperature = temperature
        self.tbsize = tbsize
        self.hvp_len = hvp_len
        self.hvp_iter = hvp_iter
        self.hvp_size = hvp_size
        self.hvp_scale = hvp_scale
        

        self.model = DIFF_no_att(model_pt=self.model_pt, selector_dims=self.selector_dims, layer_dims=self.layer_dims,
                           input_dim=self.input_dim, dropout=self.dropout, activation=self.activation,
                           l2_reg=self.l2_reg, use_bn=True)
        self.criterion = keras.losses.CategoricalCrossentropy(from_logits=False)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.metric = {'train_loss': tf.keras.metrics.Mean()}

    def update_hvp(self, train_dataset, test_dataset):
        grads = None
        for ds in test_dataset.shuffle(self.tbsize, reshuffle_each_iteration=True).batch(self.tbsize):
            with tf.GradientTape() as tpe:
                x, y = ds['x'], ds['y']
                y_pred, input_x, weight_x= self.model(x, is_training=True)
                loss = self.criterion(y, y_pred)
                grads = tpe.gradient(loss, self.model.mlp.trainable_variables)

        estimate = None
        count_i = .0
        for r in range(self.hvp_len):
            cur_estimate = grads.copy()
            count_i += 1.0

            for i in range(self.hvp_iter):
                train_dataset.subsample(size=self.hvp_size)
                hvp_data = train_dataset(subsample=True)
                hessian_vector = None
                for ds in hvp_data.shuffle(self.hvp_size, reshuffle_each_iteration=True).batch(self.hvp_size):
                    x, y = ds['x'], ds['y']
                    hessian_vector = self.iter_hvp(x, y, cur_estimate)
                cur_estimate_new = [grads[i] + 0.99 * cur_estimate[i] - hessian_vector[i] / self.hvp_scale
                                    for i in range(len(hessian_vector))]

                cur_estimate_tensor = tf.concat([tf.reshape(cur_estimate[i], (1, -1)) for i in range(len(cur_estimate))],
                                                axis=1)
                cur_estimate_new_tensor = tf.concat([tf.reshape(cur_estimate_new[i], (1, -1))
                                                     for i in range(len(cur_estimate_new))], axis=1)
                diff = (np.linalg.norm(cur_estimate_new_tensor) - np.linalg.norm(cur_estimate_tensor)
                        ) / np.linalg.norm(cur_estimate_tensor)

                if diff <= 1e-5:
                    cur_estimate = cur_estimate_new
                    break
                cur_estimate = cur_estimate_new

            if estimate is None:
                estimate = [cur_est / self.hvp_scale for cur_est in cur_estimate.copy()]
            else:
                for i in range(len(estimate)):
                    estimate[i] += cur_estimate[i] / self.hvp_scale
        return [est / count_i for est in estimate]

    def iter_hvp(self, x, y, cur_estimate):
        with tf.GradientTape() as tpe1:
            with tf.GradientTape() as tpe2:
                y_pred, _, _ = self.model(x)
                loss = self.criterion(y, y_pred)
                grads = tpe1.gradient(loss, self.model.mlp.trainable_variables)

                elemwise_products = [
                    tf.multiply(grad_elem, tf.stop_gradient(v_elem))
                    for grad_elem, v_elem in zip(grads, cur_estimate) if grad_elem is not None
                ]

                grads_with_none = tpe2.gradient(elemwise_products, self.model.mlp.trainable_variables)
                return_grads = [
                    grad_elem if grad_elem is not None else tf.zeros_like(x)
                    for x, grad_elem in zip(self.model.mlp.trainable_variables, grads_with_none)
                ]
        return return_grads

    def update_loss(self, hvp, x, y):
        with tf.GradientTape() as tpe1:
            with tf.GradientTape() as tpe2:
                tpe2.watch(x)
                y_pred, _, weight = self.model(x, self.temperature, is_training=True)
                loss = self.criterion(y, y_pred)
                grads = tpe1.gradient(loss, self.model.mlp.trainable_variables)
                influence_loss = tf.add_n(
                    [tf.reduce_sum(tf.multiply(a, tf.stop_gradient(b)))] for a, b in zip(grads, hvp)
                )
                influence_score = tpe2.gradient(influence_loss, x)
                influence_score = tf.negative(influence_score)
                weight = tf.cast(weight, dtype=tf.double)
                influence_score = influence_score * weight
        return tf.reduce_sum(influence_score)
    def train_epoch(self, train_dataset, test_dataset):  
        cout = 0
        for ds in train_dataset.shuffle(self.bsize, reshuffle_each_iteration=True).batch(self.bsize):
            x, y = ds['x'], ds['y']
            sub_ds = SubNpyDataset(np.concatenate((y, x), axis=1), y_dim=2, x_dim=self.input_dim)
            hvp = self.update_hvp(sub_ds, test_dataset)

            with tf.GradientTape() as tpe:
                loss = self.update_loss(hvp, x, y)

                variables = self.model.selector.trainable_variables# + self.model.attention.trainable_variables
                grads = tpe.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(grads, variables))
            self.metric['train_loss'].update_state(loss)
            if cout == 0:
                break

    def train(self, tr_ds, val_ds, our_model_pt):
        epoch = self.epoch
        best_epoch, best_metric = 0, 1e9
        self.save_model = None
        for i in range(epoch):
            self.metric['train_loss'].reset_states()

            self.train_epoch(tr_ds(), val_ds())

            if self.metric['train_loss'].result().numpy() < best_metric:
                best_metric = self.metric['train_loss'].result().numpy()
                best_epoch = i
                self.save_model = self.model
                #tf.saved_model.save(self.model, self.model_pt)

            if i - best_epoch > 10:
                break
        tf.saved_model.save(self.save_model, our_model_pt)
        return best_metric


class ReTrainer(object):
    def __init__(self, epoch, model_pt, activation, mlp_dim, bsize, lr, dropout, l2_reg):
        self.epoch = epoch
        self.model_pt = model_pt
        self.activation = activation
        self.layer_dims = [mlp_dim, mlp_dim]
        self.bsize = bsize
        self.lr = lr
        self.dropout = dropout
        self.l2_reg = l2_reg

        self.model = DIFF2(model_pt=self.model_pt,  layer_dims=self.layer_dims, dropout=self.dropout,
                           activation=self.activation, l2_reg=self.l2_reg, use_bn=True)
        self.criterion = keras.losses.CategoricalCrossentropy(from_logits=False)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.metric = {'test_auc': tf.keras.metrics.AUC(from_logits=False), 'train_loss': tf.keras.metrics.Mean(),
                       'test_loss': tf.keras.metrics.Mean()}

    def train_epoch(self, train_dataset, test_dataset):
        for ds in train_dataset.shuffle(self.bsize, reshuffle_each_iteration=True).batch(self.bsize):
            x, y = ds['x'], ds['y']

            with tf.GradientTape() as tpe:
                y_pred = self.model(x, is_training=True)
                loss = self.criterion(y, y_pred)
                grads = tpe.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.metric['train_loss'].update_state(loss)

        for ds in test_dataset.batch(self.bsize):
            x, y = ds['x'], ds['y']
            pred = self.model(x, is_training=False)
            loss = self.criterion(y, pred)
            self.metric['test_loss'].update_state(loss)
            self.metric['test_auc'].update_state(y_true=y[:, 1], y_pred=pred[:, 1])

    def train(self, tr_ds, val_ds, te_ds, save_model, influence_loss):
        epoch = self.epoch
        best_epoch, best_metric = 0, 0
        for i in range(epoch):
            self.metric['train_loss'].reset_states()
            self.metric['test_loss'].reset_states()
            self.metric['test_auc'].reset_states()
            self.train_epoch(tr_ds(), val_ds())
            train_loss = self.metric['train_loss'].result().numpy()
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
            pred = self.model(x, is_training=False)
            loss = self.criterion(y, pred)
            self.metric['test_loss'].update_state(loss)
            self.metric['test_auc'].update_state(y_true=y[:, 1], y_pred=pred[:, 1])
        test_auc = self.metric['test_auc'].result().numpy()
        print('best_epoch = {}, influence_loss = {}, test auc = {}'.format(best_epoch, influence_loss, test_auc))

        return best_metric


def setup_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def diff(tr_ds, val_ds, te_ds, model_pl='latents/synthetic_data_syn1/', iteration=300, seed=0, gpu_on=False,
         save_model=False, input_dim=11, n_trials=300, **unused):
    progress = WorkSplitter()

    if not os.path.exists(model_pl + 'DIFF/'):
        os.makedirs(model_pl + 'DIFF/')
    our_model_pl = model_pl + 'DIFF/'

    progress.section("DIFF: Set the random seed")
    setup_seed(seed)

    progress.section("DIFF: Training")
    tuner = Tuner()
    trials, best_params = tuner.tune(n_trials=n_trials, our_model_pl=our_model_pl, model_pl=model_pl, gpu_on=gpu_on,
                                     train=tr_ds, valid=val_ds, test=te_ds, epoch=iteration, seed=seed,
                                     save_model=save_model, input_dim=input_dim)
    return trials, best_params
