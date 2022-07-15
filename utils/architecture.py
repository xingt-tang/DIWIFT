import numpy as np
import tensorflow as tf
from keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.models import Model


class MultiLayerPerceptron(Model):
    def __init__(self, layer_dims, out_dims=2, dropout=.0, activation=None, out_activation=None, l2_reg=.0,
                 use_bn=False, output_layer=True, seed=2021, name='mlp'):
        super(MultiLayerPerceptron, self).__init__(name)
        self.dims = layer_dims
        self.dropout_rate = dropout
        if activation == 'relu':
            self.activation = keras.activations.relu
        else:
            self.activation = keras.activations.selu
        self.out_activation = out_activation
        self.l2 = l2_reg
        self.out_dims = out_dims
        self.use_bn = use_bn
        self.output_layer = output_layer
        self.seed = seed
        self.kernels = None
        self.bias = None
        self.bn = None
        self.output_kernel = None
        self.output_bias = None
        self.dropout = None

    def build(self, input_shape):
        dims = [input_shape[1]] + self.dims
        self.kernels = [self.add_weight(name='kernel_{}'.format(i), shape=(dims[i], dims[i + 1]),
                                        initializer=keras.initializers.HeUniform(seed=self.seed),
                                        regularizer=keras.regularizers.L2(self.l2), trainable=True)
                        for i in range(len(dims) - 1)]
        self.bias = [
            self.add_weight(name='bias_{}'.format(i), shape=(self.dims[i], ), initializer=keras.initializers.Zeros(),
                            regularizer=keras.regularizers.L2(self.l2), trainable=True)
            for i in range(len(self.dims))]

        if self.output_layer:
            self.output_kernel = self.add_weight(name='output_kernel', shape=(self.dims[-1], self.out_dims),
                                                 initializer=keras.initializers.HeUniform(seed=self.seed),
                                                 regularizer=keras.regularizers.L2(self.l2), trainable=True)
            self.output_bias = self.add_weight(name='output_bias', shape=(self.out_dims, ),
                                               initializer=keras.initializers.Zeros(),
                                               regularizer=keras.regularizers.L2(self.l2), trainable=True)

        if self.use_bn:
            self.bn = [keras.layers.BatchNormalization() for _ in range(len(self.dims))]

        if self.dropout_rate > .0:
            self.dropout = [keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in range(len(self.dims))]

        super(MultiLayerPerceptron, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        assert len(inputs.shape) == 2, 'inputs dims should equal 2'
        deep_outs = inputs
        for i in range(len(self.dims)):
            deep_outs = tf.matmul(deep_outs, self.kernels[i]) + self.bias[i]
            deep_outs = self.activation(deep_outs)

            if self.use_bn:
                deep_outs = self.bn[i](deep_outs, training=training)

            if self.dropout_rate > .0:
                deep_outs = self.dropout[i](deep_outs, training=training)

        if self.output_layer:
            deep_outs = tf.matmul(deep_outs, self.output_kernel) + self.output_bias

        if self.out_activation == 'softmax':
            deep_outs = tf.nn.softmax(deep_outs)
        elif self.out_activation == 'sigmoid':
            deep_outs = tf.nn.sigmoid(deep_outs)

        return deep_outs


class L2X(Model):
    def __init__(self, model_pt, selector_dims, layer_dims, input_dim, dropout=.0, activation=None, l2_reg=.0, use_bn=False,
                 output_layer=True, seed=2021, name='l2x'):
        super(L2X, self).__init__(name)
        self.layer_dims = layer_dims
        self.dropout_rate = dropout
        self.model_pt = model_pt
        if activation == 'relu':
            self.activation = keras.activations.relu
        else:
            self.activation = keras.activations.selu
        self.l2 = l2_reg
        self.use_bn = use_bn
        self.output_layer = output_layer
        self.seed = seed
        self.selector_dims = selector_dims

        self.mlp = tf.saved_model.load(model_pt)

        # self.mlp = MultiLayerPerceptron(layer_dims=self.layer_dims, dropout=self.dropout_rate,
        #                                 activation=self.activation, out_activation='softmax', use_bn=self.use_bn)
        self.selector = MultiLayerPerceptron(layer_dims=self.selector_dims, out_dims=input_dim,dropout=self.dropout_rate, activation=self.activation,
                                             out_activation='softmax', use_bn=False)

    def call(self, inputs, k=2, temperature=1, is_training=True, **kwargs):
        assert len(inputs.shape) == 2, 'inputs dims should equal 2'
        logits = self.selector(inputs, training=is_training)
        masks = self.create_mask(k, temperature, logits, is_training)
        new_inputs = inputs * masks
        deep_out = self.mlp(new_inputs, training=is_training)

        return deep_out, masks

    @staticmethod
    def create_mask(k, temperature, logits, is_training):
        # logits: [BATCH_SIZE, d]
        logits_ = K.expand_dims(logits, -2)  # [BATCH_SIZE, 1, d]

        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        uniform = tf.random.uniform(shape=(batch_size, k, d),
                                    minval=np.finfo(tf.float32.as_numpy_dtype).tiny,
                                    maxval=1.0)

        gumbel = - K.log(- K.log(uniform))
        noisy_logits = (gumbel + logits_) / temperature
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis=1)

        # Explanation Stage output.
        threshold = tf.expand_dims(tf.nn.top_k(logits, k, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        return K.in_train_phase(samples, discrete_logits, training=is_training)


class CL2X(Model):
    def __init__(self, model_pt, selector_dims, layer_dims, input_dim, dropout=.0, activation=None, l2_reg=.0,
                 use_bn=False, output_layer=True, seed=2021, name='cl2x'):
        super(CL2X, self).__init__(name)
        self.layer_dims = layer_dims
        self.dropout_rate = dropout
        if activation == 'relu':
            self.activation = keras.activations.relu
        else:
            self.activation = keras.activations.selu
        self.l2 = l2_reg
        self.use_bn = use_bn
        self.output_layer = output_layer
        self.seed = seed
        self.selector_dims = selector_dims

        self.mlp = tf.saved_model.load(model_pt)

        self.selector = MultiLayerPerceptron(layer_dims=self.selector_dims, out_dims=input_dim,
                                             dropout=self.dropout_rate, activation=self.activation,
                                             out_activation='softmax', use_bn=False)

    def call(self, inputs, k=2, temperature=1, is_training=True, **kwargs):
        assert len(inputs.shape) == 2, 'inputs dims should equal 2'
        logits = self.selector(inputs, training=is_training)
        masks = self.create_mask(k, temperature, logits, is_training)
        new_inputs = inputs * masks
        deep_out = self.mlp(new_inputs, training=is_training)
        ori_deep_out = tf.stop_gradient(self.mlp(inputs, training=is_training))

        return deep_out, ori_deep_out, masks

    @staticmethod
    def create_mask(k, temperature, logits, is_training):
        # logits: [BATCH_SIZE, d]
        logits_ = K.expand_dims(logits, -2)  # [BATCH_SIZE, 1, d]

        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        uniform = tf.random.uniform(shape=(batch_size, k, d),
                                    minval=np.finfo(tf.float32.as_numpy_dtype).tiny,
                                    maxval=1.0)

        gumbel = - K.log(- K.log(uniform))
        noisy_logits = (gumbel + logits_) / temperature
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis=1)

        # Explanation Stage output.
        threshold = tf.expand_dims(tf.nn.top_k(logits, k, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        return K.in_train_phase(samples, discrete_logits, training=is_training)


class INVASE(Model):
    def __init__(self, model_pt, selector_dims, layer_dims, critic_dims, input_dim, dropout=.0, activation=None, l2_reg=.0, use_bn=False,
                 output_layer=True, seed=2021, name='invase'):
        super(INVASE, self).__init__(name)
        self.layer_dims = layer_dims
        self.model_pt = model_pt
        self.dropout_rate = dropout
        if activation == 'relu':
            self.activation = keras.activations.relu
        else:
            self.activation = keras.activations.selu
        self.l2 = l2_reg
        self.use_bn = use_bn
        self.output_layer = output_layer
        self.seed = seed
        self.selector_dims = selector_dims
        self.critic_dims = critic_dims

        #self.mlp = tf.saved_model.load(model_pt)

        self.mlp = MultiLayerPerceptron(layer_dims=self.layer_dims, dropout=self.dropout_rate,
                                        activation=self.activation, out_activation='softmax', use_bn=use_bn)
        self.actor = MultiLayerPerceptron(layer_dims=self.selector_dims, out_dims=input_dim, dropout=self.dropout_rate,
                                          activation=self.activation, out_activation='sigmoid', use_bn=False)
        self.critic = MultiLayerPerceptron(layer_dims=self.critic_dims, dropout=self.dropout_rate,
                                           activation=self.activation, out_activation='softmax', use_bn=use_bn)

    @tf.function
    def call(self, inputs, is_training=True, **kwargs):
        assert len(inputs.shape) == 2, 'inputs dims should equal 2'
        logits = self.actor(inputs, training=is_training)
        masks = self.create_mask(logits)
        new_inputs = masks * inputs
        deep_out = self.critic(new_inputs, training=is_training)
        ori_deep_out = self.mlp(inputs, training=is_training)

        return deep_out, ori_deep_out, logits, masks

    @staticmethod
    def create_mask(prob):
        threshold = tf.random.uniform(shape=prob.shape)
        masks = tf.cast(tf.less_equal(threshold, prob), tf.float32)

        return masks


class DIFF1(Model):
    def __init__(self, model_pt, selector_dims, layer_dims, input_dim, dropout=.0, heads=1, activation=None, l2_reg=.0,
                 use_bn=False, output_layer=True, seed=2021, name='diff1'):
        super(DIFF1, self).__init__(name)
        self.layer_dims = layer_dims
        self.dropout_rate = dropout
        if activation == 'relu':
            self.activation = keras.activations.relu
        else:
            self.activation = keras.activations.selu
        self.l2 = l2_reg
        self.use_bn = use_bn
        self.output_layer = output_layer
        self.seed = seed
        self.selector_dims = selector_dims
        self.heads = heads

        self.mlp = tf.saved_model.load(model_pt)
        self.attention = keras.layers.MultiHeadAttention(num_heads=heads, key_dim=1)
        self.selector = MultiLayerPerceptron(layer_dims=self.selector_dims, out_dims=input_dim,
                                             dropout=self.dropout_rate, activation=self.activation,
                                             out_activation=None, l2_reg=self.l2, use_bn=False)

    @tf.function
    def call(self, inputs, temperature=1, is_training=True, **kwargs):
        assert len(inputs.shape) == 2, 'inputs dims should equal 2'
        inputs_init = inputs
        inputs = tf.expand_dims(inputs, axis=-1)
        outputs, scores = self.attention(inputs, inputs, return_attention_scores=True)
        deep_outs = tf.squeeze(outputs, axis=-1)

        deep_outs = self.selector(deep_outs, training=is_training)

        feature_weights = tf.nn.sigmoid(deep_outs / temperature)
        # feature_weights = self.mean_one_normalization(feature_weights)

        mlp_inputs = inputs_init * feature_weights
        mlp_outputs = self.mlp(mlp_inputs, training=is_training)
        return mlp_outputs, mlp_inputs, feature_weights


class DIFF_no_att(Model):
    def __init__(self, model_pt, selector_dims, layer_dims, input_dim, dropout=.0, heads=1, activation=None, l2_reg=.0,
                 use_bn=False, output_layer=True, seed=2021, name='diff1'):
        super(DIFF_no_att, self).__init__(name)
        self.layer_dims = layer_dims
        self.dropout_rate = dropout
        if activation == 'relu':
            self.activation = keras.activations.relu
        else:
            self.activation = keras.activations.selu
        self.l2 = l2_reg
        self.use_bn = use_bn
        self.output_layer = output_layer
        self.seed = seed
        self.selector_dims = selector_dims
        self.heads = heads

        self.mlp = tf.saved_model.load(model_pt)
        #self.attention = keras.layers.MultiHeadAttention(num_heads=heads, key_dim=1)
        self.selector = MultiLayerPerceptron(layer_dims=self.selector_dims, out_dims=input_dim,
                                             dropout=self.dropout_rate, activation=self.activation,
                                             out_activation=None, l2_reg=self.l2, use_bn=False)

    @tf.function
    def call(self, inputs, temperature=1, is_training=True, **kwargs):
        assert len(inputs.shape) == 2, 'inputs dims should equal 2'
        inputs_init = inputs
        #inputs = tf.expand_dims(inputs, axis=-1)
        #outputs, scores = self.attention(inputs, inputs, return_attention_scores=True)
        #deep_outs = tf.squeeze(outputs, axis=-1)

        deep_outs = self.selector(inputs, training=is_training)

        feature_weights = tf.nn.sigmoid(deep_outs / temperature)

        mlp_inputs = inputs_init * feature_weights
        mlp_outputs = self.mlp(mlp_inputs, training=is_training)
        return mlp_outputs, mlp_inputs, feature_weights

class DIFF_no_influ(Model):
    def __init__(self, model_pt, selector_dims, layer_dims, input_dim, dropout=.0, heads=1, activation=None, l2_reg=.0,
                 use_bn=False, output_layer=True, seed=2021, name='diff1'):
        super(DIFF_no_influ, self).__init__(name)
        self.layer_dims = layer_dims
        self.dropout_rate = dropout
        if activation == 'relu':
            self.activation = keras.activations.relu
        else:
            self.activation = keras.activations.selu
        self.l2 = l2_reg
        self.use_bn = use_bn
        self.output_layer = output_layer
        self.seed = seed
        self.selector_dims = selector_dims
        self.heads = heads

        self.mlp = MultiLayerPerceptron(layer_dims=self.layer_dims, dropout=self.dropout_rate,
                                        activation=self.activation, out_activation='softmax', l2_reg=self.l2,
                                        use_bn=self.use_bn)
        self.attention = keras.layers.MultiHeadAttention(num_heads=heads, key_dim=1)
        self.selector = MultiLayerPerceptron(layer_dims=self.selector_dims, out_dims=input_dim,
                                             dropout=self.dropout_rate, activation=self.activation,
                                             out_activation=None, l2_reg=self.l2, use_bn=False)

    @tf.function
    def call(self, inputs, temperature=1, is_training=True, **kwargs):
        assert len(inputs.shape) == 2, 'inputs dims should equal 2'
        inputs_init = inputs
        inputs = tf.expand_dims(inputs, axis=-1)
        outputs, scores = self.attention(inputs, inputs, return_attention_scores=True)
        deep_outs = tf.squeeze(outputs, axis=-1)

        deep_outs = self.selector(deep_outs, training=is_training)

        feature_weights = tf.nn.sigmoid(deep_outs / temperature)
        # feature_weights = self.mean_one_normalization(feature_weights)

        mlp_inputs = inputs_init * feature_weights
        mlp_outputs = self.mlp(mlp_inputs, training=is_training)
        return mlp_outputs
        

class DIFF2(Model):
    def __init__(self, model_pt, layer_dims, dropout=.0, heads=1, activation=None, l2_reg=.0, use_bn=False,
                 output_layer=True, seed=2021, name='diff2'):
        super(DIFF2, self).__init__(name)
        self.layer_dims = layer_dims
        self.dropout_rate = dropout
        if activation == 'relu':
            self.activation = keras.activations.relu
        else:
            self.activation = keras.activations.selu
        self.l2 = l2_reg
        self.use_bn = use_bn
        self.output_layer = output_layer
        self.seed = seed
        self.heads = heads

        self.attention = tf.saved_model.load(model_pt)
        self.mlp = MultiLayerPerceptron(layer_dims=self.layer_dims, dropout=self.dropout_rate,
                                        activation=self.activation, out_activation='softmax', l2_reg=self.l2,
                                        use_bn=self.use_bn)

    @tf.function
    def call(self, inputs, is_training=True, **kwargs):
        assert len(inputs.shape) == 2, 'inputs dims should equal 2'
        _, mlp_inputs, _ = self.attention(inputs, training=False)
        mlp_outputs = self.mlp(mlp_inputs, training=is_training)
        return mlp_outputs

