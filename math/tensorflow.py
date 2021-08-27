import numpy as np
import tensorflow as tf
from tensorflow import math as tf_math


# psnr: 其值不能很好地反映人眼主觀感受
def psnr(y_label, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    _result = tf_math.subtract(y_label, y_pred)
    _result = tf_math.square(_result)
    _result = tf_math.reduce_mean(_result)
    _result = tf_math.multiply(-10., log(_result, 10.))
    return _result


def multiOperation(op, *args):
    result = args[0]
    length = len(args)

    for i in range(1, length):
        result = op(result, args[i])

    return result


def log(x, base=None):
    if base is None:
        return tf.math.log(x)
    else:
        return tf_math.divide(tf.math.log(x), tf.math.log(base))


if __name__ == "__main__":
    log_e = log(np.e)
    log_10 = log(100., 10.)
    log_2 = log(8., 2.)
    with tf.Session() as sess:
        print("log_e:", sess.run(log_e))
        print("log_10:", sess.run(log_10))
        print("log_2:", sess.run(log_2))
