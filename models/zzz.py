import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

from utils.progress import WorkSplitter
from utils.architecture import MultiLayerPerceptron, FeatureSelection


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
        if self.save_model:
            if self.model_pl.endswith('coat/'):
                mlp_dim = trial.suggest_categorical('rank', [200])
                bsize = trial.suggest_categorical('batch_size', [512])
                lr = trial.suggest_categorical('learning_rate', [0.001])
                dropout = trial.suggest_categorical('dropout', [0.0])
                l2_reg = trial.suggest_categorical('l2_reg', [0.01])
            elif self.model_pl.endswith('adult/'):
                mlp_dim = trial.suggest_categorical('rank', [200])
                bsize = trial.suggest_categorical('batch_size', [512])
                lr = trial.suggest_categorical('learning_rate', [0.001])
                dropout = trial.suggest_categorical('dropout', [0.0])
                l2_reg = trial.suggest_categorical('l2_reg', [0.01])

        else:
            mlp_dim = trial.suggest_categorical('rank', [50, 100, 150, 200])
            bsize = trial.suggest_categorical('batch_size', [128, 256, 512, 1024, 2048])
            test_bsize = trial.suggest_categorical('test_bsize', [2048])
            lr = trial.suggest_categorical('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1])
            l2_reg = trial.suggest_categorical('lambda', [0.00001, 0.0001, 0.001, 0.01, 0.1])
            dropout = trial.suggest_categorical('dropout', [0.0])
            hvp_iter = trial.suggest_categorical('hvp_iter', [1000])
            hvp_len = trial.suggest_categorical('hvp_len', [1])
            hvp_size = trial.suggest_categorical('hvp_size', [100])
            hvp_scale = trial.suggest_categorical('hvp_scale', [200])


        setup_seed(self.seed)

        if self.model_pl.endswith('syn1/'):
            activation = 'relu'
        else:
            activation = 'selu'

        model = Trainer(self.epoch, self.model_pl, activation, mlp_dim, self.input_dim, bsize, test_bsize, lr, dropout, l2_reg, hvp_iter, hvp_len, hvp_size, hvp_scale)
        score = model.train(self.train, self.valid, self.test, self.save_model)

        return score


class Tuner:
    """Class for tuning hyperparameter of MF models."""

    def __init__(self):
        """Initialize Class."""

    def tune(self, n_trials, model_pl, gpu_on, train, valid, test, epoch, seed, save_model,input_dim):
        """Hyperparameter Tuning by TPE."""
        objective = Objective(model_pl=model_pl, gpu_on=gpu_on, train=train, valid=valid, test=test, epoch=epoch,
                              seed=seed, save_model=save_model, input_dim=input_dim)
        study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.trials_dataframe(), study.best_params


class Trainer(object):
    def __init__(self, epoch, model_pt, activation, mlp_dim, input_dim, bsize, test_bsize, lr, dropout, l2_reg, hvp_iter, hvp_len, hvp_size, hvp_scale):
        self.layer_dims = [mlp_dim, mlp_dim]
        self.input_dim = input_dim
        self.model_pt = model_pt
        #self.FWI_model_pt = FWI_model_pt

        self.hvp_iter = hvp_iter
        self.hvp_len = hvp_len
        self.hvp_size = hvp_size
        self.hvp_scale = hvp_scale
        self.epoch = epoch
        self.lr = lr
        self.bsize = bsize
        self.test_bsize = test_bsize
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.activation = activation

        self.model = FeatureSelection(model_pt=self.model_pt, layer_dims=self.layer_dims, input_dim=self.input_dim,
                                      heads=1)
        self.criterion = keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.metric = {"test_auc": tf.keras.metrics.AUC(), "train_loss": tf.keras.metrics.Mean(),
                       "test_loss": tf.keras.metrics.Mean()}

    def update_hvp(self, tr, va):
        grads = None
        for ds in va.shuffle(self.test_bsize, reshuffle_each_iteration=True).batch(self.test_bsize):
            with tf.GradientTape() as tpe:
                x, y = ds["x"], ds["y"]
                y_hat, _, weight = self.model(x)
                loss = self.criterion(y_hat, y)
                grads = tpe.gradient(loss, self.model.mlp.trainable_variables)

        estimate = None
        count_i = .0
        for r in range(self.hvp_len):
            cur_estimate = grads.copy()
            count_i += 1.0
            for i in range(self.hvp_iter):
                tr.subsample(size=self.hvp_size)
                hvp_data = tr(subsample=True)
                for ds in hvp_data.shuffle(self.hvp_size, reshuffle_each_iteration=True).batch(self.hvp_size):
                    x, y = ds["x"], ds["y"]
                    hessian_vector = self.iter_hvp(x, y, grads, cur_estimate)
                cur_estimate_new = [grads[i] + 0.99 * cur_estimate[i] - hessian_vector[i]/self.hvp_scale for i in range(len(hessian_vector))]

                cur_estimate_tensor = tf.concat([tf.reshape(cur_estimate[i], (1, -1)) for i in range(len(cur_estimate))], axis=1)
                cur_estimate_new_tensor = tf.concat([tf.reshape(cur_estimate_new[i], (1, -1)) for i in range(len(cur_estimate_new))], axis=1)
                diff = (np.linalg.norm(cur_estimate_new_tensor) - np.linalg.norm(cur_estimate_tensor)) / np.linalg.norm(cur_estimate_tensor)

                if diff <= 1e-5:
                    #print("satisfy threshold !!!")
                    cur_estimate = cur_estimate_new
                    break
                cur_estimate = cur_estimate_new

            #print("repeat {}, diff is {}, cur_estimate_new = {}, cur_estimate = {}".format(count_i, diff, np.linalg.norm(cur_estimate_new_tensor), np.linalg.norm(cur_estimate_tensor)))

            if estimate is None:
                estimate = [cur_est/self.hvp_scale for cur_est in cur_estimate.copy()]
            else:
                for j in range(len(estimate)):
                    estimate[j] += cur_estimate[j]/self.hvp_scale
        return [est/count_i for est in estimate]

    def iter_hvp(self, x, y, grads, cur_estimate):
        with tf.GradientTape() as tpe1:
            with  tf.GradientTape() as tpe2:
                y_hat, _, _ = self.model(x)
                loss = self.criterion(y_hat, y)
                grads = tpe1.gradient(loss, self.model.mlp.trainable_variables)

                #print(grads)
                elemwise_products = [
                    tf.multiply(grad_elem,tf.stop_gradient(v_elem))
                    for grad_elem, v_elem in zip(grads, cur_estimate) if grad_elem is not None
                ]
                #print(elemwise_products)
                grads_with_none = tpe2.gradient(elemwise_products, self.model.mlp.trainable_variables)
                return_grads = [
                    grad_elem if grad_elem is not None \
                        else tf.zeros_like(x) \
                    for x, grad_elem in zip(self.model.mlp.trainable_variables, grads_with_none)]
        return return_grads

    def update_loss(self, hvp, x, y, tem):
        with tf.GradientTape() as tp1:
            with tf.GradientTape() as tp2:
                tp2.watch(x)
                y_hat, _, weight = self.model(x, tem)
                loss = self.criterion(y, y_hat)
                grads = tp1.gradient(loss, self.model.mlp.trainable_variables)
                influence_loss = tf.constant(0.0, dtype=tf.float32)
                tp2.watch(influence_loss)
                influence_loss = tf.add_n(
                    [tf.reduce_sum(tf.multiply(a, tf.stop_gradient(b))) for a, b in zip(grads, hvp)])
                # for grad, hvp_grad in zip(grads, hvp):
                #     influence_loss += tf.reduce_sum(grad * tf.stop_gradient(hvp_grad))
                influence_score = tp2.gradient(influence_loss, x)
                influence_score = tf.negative(influence_score)
                influence_score = influence_score * weight
        return tf.reduce_sum(influence_score)

        #influence_score_sum = tf.reduce_sum(tf.abs(influence_score), axis=1)
        # tf.negative(tf.reduce_sum(influence_score_sum))


    def train_epoch(self, tr_ds, val_ds, te_ds):
        influence_loss = []
        best_metric = 1e4
        best_epoch = 0
        for i in range(self.epoch):

            tem = 1 #max(0.001, 1-0.999 * i / 60)
            self.metric["train_loss"].reset_states()
            for ds in tr_ds.shuffle(self.bsize, reshuffle_each_iteration=True).batch(self.bsize):
                hvp = self.update_hvp(tr_ds, val_ds)
                x, y = ds['x'], ds['y']
                with tf.GradientTape() as tpe:
                    loss = self.update_loss(hvp, x, y, tem)
                    variables = self.model.selector.trainable_variables + self.model.attention.trainable_variables
                    grads = tpe.gradient(loss, variables)
                    self.optimizer.apply_gradients(zip(grads, variables))
                    self.metric["train_loss"].update_state(loss)
            train_loss = self.metric["train_loss"].result().numpy()

            influence_loss.append(train_loss)

            if train_loss < best_metric:
                best_metric = train_loss
                best_epoch = i
            if i - best_epoch > 10:
                break
        print("best_epoch = {}, lr = {}, bsize = {}, mlp_dims = {}, best loss = {}".format(best_epoch, self.lr, self.bsize, self.mlp_dim, best_metric))
        return best_metric


def setup_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def diff(tr_ds, val_ds, te_ds, model_pl='latents/synthetic_data_syn1/', iteration=300, seed=0, gpu_on=False,
         save_model=False, input_dim=11, n_trials=200, **unused):
    progress = WorkSplitter()

    progress.section("DIFF: Set the random seed")
    setup_seed(seed)

    progress.section("DIFF: Training")
    tuner = Tuner()
    trials, best_params = tuner.tune(n_trials=n_trials, model_pl=model_pl, gpu_on=gpu_on, train=tr_ds, valid=val_ds,
                                     test=te_ds, epoch=iteration, seed=seed, save_model=save_model, input_dim=input_dim)
    return trials, best_params
