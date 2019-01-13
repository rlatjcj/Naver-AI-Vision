from functools import wraps
import tensorflow as tf
import keras.backend as K

def preprocess(y_true, y_pred):
    s = K.int_shape(y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    # y_pred = K.one_hot(K.argmax(y_pred), s[-1])
    y_pred = K.cast(y_pred > 0.5, dtype=y_pred.dtype)

    return y_true, y_pred

def as_keras_metric(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        value, update_op = method(self, *args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

@as_keras_metric
def precision(y_true, y_pred):
    y_true, y_pred = preprocess(y_true, y_pred)
    return tf.metrics.precision(y_true, y_pred)

@as_keras_metric
def recall(y_true, y_pred):
    y_true, y_pred = preprocess(y_true, y_pred)
    return tf.metrics.recall(y_true, y_pred)

def f1(y_true, y_pred):
    _recall = recall(y_true, y_pred)
    _precision = precision(y_true, y_pred)
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())
    return _f1score