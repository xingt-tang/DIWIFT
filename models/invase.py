import numpy as np
import tensorflow as tf
from keras import backend as K
import tensorflow.keras as keras

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

from utils.architecture import INVASE
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
        mlp_dim = trial.suggest_categorical('mlp_rank', [50, 100, 150, 200]) #[50, 100, 150, 200]
        selector_dim = trial.suggest_categorical('selector_rank', [50, 100, 150, 200])
        critic_dim = trial.suggest_categorical('critic_rank', [50, 100, 150, 200])
        bsize = trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096])
        lr = trial.suggest_categorical('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1])
        l2_reg = trial.suggest_categorical('lambda', [0.001])
        dropout = trial.suggest_categorical('dropout', [0.0])

        if self.model_pl.endswith('syn1/'):
            beta = trial.suggest_categorical('beta', [0.1])
        elif self.model_pl.endswith('syn3/'):
            beta = trial.suggest_categorical('beta', [0.1])
        else:
            beta = trial.suggest_categorical('beta', [0.001])

        setup_seed(self.seed)

        if self.model_pl.endswith('syn1/'):
            activation = 'relu'
        else:
            activation = 'relu'

        model = Trainer(self.epoch, self.model_pl, activation, mlp_dim, selector_dim, critic_dim, self.input_dim, bsize, lr,
                        dropout, l2_reg, beta)
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
    def __init__(self, epoch, model_pt, activation, mlp_dim, selector_dim, critic_dim, input_dim, bsize, lr, dropout, l2_reg, beta):
        self.epoch = epoch
        self.model_pt = model_pt
        self.activation = activation
        self.layer_dims = [mlp_dim, mlp_dim]
        self.bsize = bsize
        self.lr = lr
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.beta = beta
        self.selector_dims = [selector_dim, selector_dim]
        self.input_dim = input_dim
        self.critic_dims = [critic_dim, critic_dim]
        
        print("invase pretrain model path : ",self.model_pt)

        self.model = INVASE(model_pt=self.model_pt, selector_dims=self.selector_dims, layer_dims=self.layer_dims, critic_dims = self.critic_dims, input_dim=self.input_dim,
                            dropout=self.dropout, activation=self.activation, l2_reg=self.l2_reg, use_bn=True)
        self.criterion = keras.losses.CategoricalCrossentropy(from_logits=False)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.metric = {'test_auc': tf.keras.metrics.AUC(from_logits=False), 'train_loss': tf.keras.metrics.Mean(),
                       'test_loss': tf.keras.metrics.Mean()}

    def actor_loss(self, masks, critic_out, baseline_out, y_true, actor_out):
        y_true = tf.cast(y_true, dtype=tf.float32)
        critic_loss = -tf.reduce_sum(y_true * tf.math.log(critic_out + 1e-8), axis=1)

        baseline_loss = -tf.reduce_sum(y_true * tf.math.log(baseline_out + 1e-8), axis=1)
        Reward = -(critic_loss - baseline_loss)

        custom_actor_loss = Reward * tf.reduce_sum(
            masks * K.log(actor_out + 1e-8) + (1 - masks) * K.log(
                1 - actor_out + 1e-8), axis=1) - self.beta * tf.reduce_mean(actor_out, axis=1)

        custom_actor_loss = tf.reduce_mean(-custom_actor_loss)

        return custom_actor_loss

    def train_epoch(self, train_dataset, test_dataset):
        for ds in train_dataset.shuffle(self.bsize, reshuffle_each_iteration=True).batch(self.bsize):
            x, y = ds['x'], ds['y']
            with tf.GradientTape() as tpe1:
                with tf.GradientTape() as tpe2:
                    with tf.GradientTape() as tpe3:
                        y_pred, ori_y_pred, x_pred, masks = self.model(x, is_training=True)
                        critic_loss = self.criterion(y, y_pred)
                        grads = tpe1.gradient(critic_loss, self.model.critic.trainable_variables)
                        self.optimizer.apply_gradients(zip(grads, self.model.critic.trainable_variables))

                        base_loss = self.criterion(y, ori_y_pred)
                        grads = tpe2.gradient(base_loss, self.model.mlp.trainable_variables)
                        self.optimizer.apply_gradients(zip(grads, self.model.mlp.trainable_variables))

                        actor_loss = self.actor_loss(masks, y_pred, ori_y_pred, y, x_pred)
                        grads = tpe3.gradient(actor_loss, self.model.actor.trainable_variables)
                        self.optimizer.apply_gradients(zip(grads, self.model.actor.trainable_variables))

            self.metric['train_loss'].update_state(actor_loss)

        for ds in test_dataset.batch(self.bsize):
            x, y = ds['x'], ds['y']
            pred, _, _, _ = self.model(x, is_training=False)
            loss = self.criterion(y, pred)
            self.metric['test_loss'].update_state(loss)
            self.metric['test_auc'].update_state(y_true=y[:, 1], y_pred=pred[:, 1])

    def train(self, tr_ds, val_ds, te_ds, save_model):
        epoch = self.epoch
        best_epoch, best_metric = 0, 100000
        for i in range(epoch):
            self.metric['train_loss'].reset_states()
            self.metric['test_loss'].reset_states()
            self.metric['test_auc'].reset_states()
            self.train_epoch(tr_ds(), val_ds())
            train_loss = self.metric['train_loss'].result().numpy()
            # print('[Epoch {}]: train loss:{}, test loss:{}, test AUC:{}'.
            #       format(i, train_loss, self.metric['test_loss'].result().numpy(),
            #              self.metric['test_auc'].result().numpy()))

            if self.metric['test_loss'].result().numpy() < best_metric:
                best_metric = self.metric['test_loss'].result().numpy()
                best_epoch = i
                if save_model:
                    tf.saved_model.save(self.model, self.model_pt)

            if i - best_epoch > 10:
                break

        self.metric['test_loss'].reset_states()
        self.metric['test_auc'].reset_states()
        for ds in te_ds().batch(self.bsize):
            x, y = ds['x'], ds['y']
            pred, _, _, _ = self.model(x, is_training=False)
            loss = self.criterion(y, pred)
            self.metric['test_loss'].update_state(loss)
            self.metric['test_auc'].update_state(y_true=y[:, 1], y_pred=pred[:, 1])
        test_auc = self.metric['test_auc'].result().numpy()
        print('best_epoch = {}, lr = {}, bsize = {}, dropout = {}, l2_reg = {}, mlp_dims = {}, selector_dim = {}, '
              'beta = {}, test auc = {}'.
              format(best_epoch, self.lr, self.bsize, self.dropout, self.l2_reg, self.layer_dims, self.selector_dims,
                     self.beta, test_auc))

        return best_metric


def setup_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def invase(tr_ds, val_ds, te_ds, model_pl='latents/synthetic_data_syn1/', iteration=300, seed=0, gpu_on=False,
           save_model=False, input_dim=11, n_trials=200, **unused):
    progress = WorkSplitter()

    progress.section("INVASE: Set the random seed")
    setup_seed(seed)

    progress.section("INVASE: Training")
    tuner = Tuner()
    trials, best_params = tuner.tune(n_trials=n_trials, model_pl=model_pl, gpu_on=gpu_on, train=tr_ds, valid=val_ds,
                                     test=te_ds, epoch=iteration, seed=seed, save_model=save_model,
                                     input_dim=input_dim)
    return trials, best_params
