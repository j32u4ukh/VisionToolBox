import numpy as np
import tensorflow as tf
from tensorflow.math import divide


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
        return divide(tf.math.log(x), tf.math.log(base))


if __name__ == "__main__":
    log_e = log(np.e)
    log_10 = log(100., 10.)
    log_2 = log(8., 2.)
    with tf.Session() as sess:
        print("log_e:", sess.run(log_e))
        print("log_10:", sess.run(log_10))
        print("log_2:", sess.run(log_2))
