import cv2
import numpy as np
import tensorflow as tf
from tensorflow.math import (
    greater,
    add,
    subtract,
    multiply,
    divide,
    square,
    pow as tf_pow,
    reduce_mean as tf_mean,
    reduce_std as tf_std
)

from math import (
    log,
    multiOperation
)


def psnr(y_label, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    _result = subtract(y_label, y_pred)
    _result = square(_result)
    _result = tf_mean(_result)
    _result = multiply(-10., log(_result, 10.))
    return _result


def tf_ssim(x, y, is_normalized=False):
    """
    k1 = 0.01
    k2 = 0.03
    L = 1.0 if is_normalized else 255.0
    c1 = np.power(k1 * L, 2)
    c2 = np.power(k2 * L, 2)
    c3 = c2 / 2
    """
    k1 = 0.01
    k2 = 0.03
    L = 1.0 if is_normalized else 255.0
    c1 = tf_pow(multiply(k1, L), 2.0)
    c2 = tf_pow(multiply(k2, L), 2.0)
    c3 = divide(c2, 2.0)

    # if type(x) is np.ndarray:
    #      x = tf.convert_to_tensor(x, dtype=tf.float32)
    # if type(y) is np.ndarray:
    #      y = tf.convert_to_tensor(y, dtype=tf.float32)

    """
    ux = x.mean()
    uy = y.mean()
    """
    ux = tf_mean(x)
    uy = tf_mean(y)

    """
    std_x = x.std()
    std_y = y.std()
    """
    std_x = tf_std(x)
    std_y = tf_std(y)

    """
    xy = (x - ux) * (y - uy)
    std_xy = xy.mean()
    """
    xy = multiply(subtract(x, ux), subtract(y, uy))
    std_xy = tf_mean(xy)

    """
    l_xy = (2 * ux * uy + c1) / (np.power(ux, 2) + np.power(uy, 2) + c1)
    """
    l_son = add(multiOperation(multiply, 2.0, ux, uy), c1)
    l_mom = multiOperation(add, tf_pow(ux, 2.0), tf_pow(uy, 2.0), c1)
    l_xy = divide(l_son, l_mom)

    """
    c_xy = (2 * std_x * std_y + c2) / (np.power(std_x, 2) + np.power(std_y, 2) + c2)
    """
    c_son = add(multiOperation(multiply, 2.0, std_x, std_y), c2)
    c_mom = multiOperation(add, tf_pow(std_x, 2.0), tf_pow(std_y, 2.0), c2)
    c_xy = divide(c_son, c_mom)

    """
    s_xy = (std_xy + c3) / (std_x * std_y + c3)
    """
    s_son = add(std_xy, c3)
    s_mom = add(multiply(std_x, std_y), c3)
    s_xy = divide(s_son, s_mom)

    one = tf.constant(1.0)
    _ssim = multiOperation(multiply, l_xy, c_xy, s_xy)
    _result = tf.cond(greater(_ssim, one), lambda: one, lambda: _ssim)

    return _result


def tf_ssim3(x, y, is_normalized=True):
    [x1, x2, x3] = tf.split(x, 3, axis=2)
    [y1, y2, y3] = tf.split(y, 3, axis=2)

    s1 = tf_ssim(x1, y1, is_normalized)
    s2 = tf_ssim(x2, y2, is_normalized)
    s3 = tf_ssim(x3, y3, is_normalized)

    three = tf.constant(3.0)
    result = divide(multiOperation(add, s1, s2, s3), three)

    return result


def tf_ssim3_(xy):
    x, y = tf.split(xy, 2, axis=3)
    x = tf.squeeze(x)
    y = tf.squeeze(y)
    return tf_ssim3(x, y, is_normalized=False)


def tf_ssim3_norm(xy):
    x, y = tf.split(xy, 2, axis=3)
    x = tf.squeeze(x)
    y = tf.squeeze(y)
    return tf_ssim3(x, y, is_normalized=True)


def tf_ssim4(x, y, is_normalized=False):
    stack = tf.stack([x, y], axis=4)

    if is_normalized:
        each_loss = tf.map_fn(tf_ssim3_norm, stack)
    else:
        each_loss = tf.map_fn(tf_ssim3_, stack)

    total_loss = tf_mean(each_loss)

    return total_loss, each_loss


if __name__ == "__main__":
    def ssimTest(img_x, img_y):
        x = tf.placeholder(dtype=tf.float32,
                           shape=[None, None, None, 3])
        y = tf.placeholder(dtype=tf.float32,
                           shape=[None, None, None, 3])
        comput_ssim = tf_ssim4(x, y, True)

        # ssim(self.labels, self.pred, is_normalized=True)
        # labels[i].shape = (None, label_size, label_size, c_dim)
        # pred: (?, 32, 32, self.c_dim)
        tf_config = tf.ConfigProto(log_device_placement=True)
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
        with tf.Session(config=tf_config) as sess:
            # ssim_value = sess.run(comput_ssim,
            #                       feed_dict={x: img_x,
            #                                  y: img_y})
            # print("result:", ssim_value)

            total_loss, each_loss = sess.run(comput_ssim,
                                             feed_dict={x: img_x,
                                                        y: img_y})

            print("total_loss:", total_loss)
            print("each_loss:", each_loss)


    # ================================================================================
    img1 = cv2.imread("data/splice1.png")
    img2 = cv2.imread("data/splice2.png")
    img3 = cv2.imread("data/splice3.png")
    img4 = cv2.imread("data/splice4.png")
    images = [img1, img2, img3, img4]
    dsts = []

    for i in range(len(images)):
        img = images[i] / 255.0
        # INTER_CUBIC = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_CUBIC)
        images[i] = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        dst = cv2.GaussianBlur(images[i].copy(), (13, 13), 0)
        dsts.append(dst)

    images = np.array(images)
    dsts = np.array(dsts)

    ssimTest(images, dsts)

    print("ssim loss:", ssim(images, dsts, is_normalized=True))

    ssim3_loss = 0
    for img, dst in zip(images, dsts):
        ssim3_loss += ssim3(img, dst, is_normalized=True)
    ssim3_loss /= len(images)
    print("ssim3_loss:", ssim3_loss)
    print("ssim4:", ssim4(images, dsts, is_normalized=True))
