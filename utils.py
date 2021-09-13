from utils import math

import cv2
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
import seaborn as sns
from keras.utils import np_utils


def preprocess(_data, _label):
    _processed_data = _data.astype('float32') / 255.0
    _onehot_label = np_utils.to_categorical(_label)

    return _processed_data, _onehot_label


class Cifar10:
    def __init__(self):
        self.label_dict = {0: "airplae",
                           1: "automobile",
                           2: "bird",
                           3: "cat",
                           4: "deer",
                           5: "dog",
                           6: "frog",
                           7: "horse",
                           8: "ship",
                           9: "truck"}

    def syuRuI(self):
        print(self.label_dict)

    def load_data(self, SyuRuI=None):
        """讀入數據"""
        if SyuRuI is None:
            SyuRuI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        x = None
        y = None
        for sri in SyuRuI:
            images = "./data/Cifar10/images/%s.npy" % self.label_dict[sri]
            images_data = np.load(images)
            if x is None:
                x = images_data
            else:
                x = np.concatenate((x, images_data), axis=0)

            labels = "./data/Cifar10/labels/%s.npy" % self.label_dict[sri]
            labels_data = np.load(labels)
            if y is None:
                y = labels_data
            else:
                y = np.concatenate((y, labels_data), axis=0)

        return x, y

    def shuffleData(self, x, y):
        '''打亂數據'''
        # 生成和數據長度相同的標籤
        indexs = np.arange(x.shape[0])
        # 打亂標籤
        np.random.shuffle(indexs)

        # 以相同順序存入新變數
        return x[indexs], y[indexs]

    def splitTrainTest(self, x, y, test=0.2):
        """劃分訓練與測試數據"""
        length = len(x)
        x_train = x[:int((1 - test) * length)]
        y_train = x[:int((1 - test) * length)]
        x_test = x[-int(test * length):]
        y_test = x[-int(test * length):]

        return (x_train, y_train), (x_test, y_test)


# for Cifar10
def plotImagesLabelsPrediction(images, labels, prediction, label_name, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    if num > 25:
        num = 25

    for i in range(num):
        ax = plt.subplot(math.ceil(num / 5), 5, i + 1)
        ax.imshow(images[idx + i])
        title = str(i) + ", " + label_name[labels[i][0]]
        if len(prediction) > 0:
            title += "=>" + label_name[prediction[i]]

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# for MNIST
def plotLabelsAndPrediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    if num > 25:
        num = 25

    for i in range(num):
        ax = plt.subplot(math.ceil(num / 5), 5, i + 1)
        ax.imshow(np.reshape(images[idx + i], (28, 28)), cmap='binary')
        title = "label = " + str(np.argmax(labels[idx + i]))
        if len(prediction) > 0:
            title += ", predict = " + str(prediction[idx + i])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def showSingleColor(r, b, g):
    height, width = 300, 300
    img = np.zeros((height, width, 3), np.uint8)
    for h in range(height):
        for w in range(width):
            img[h, w] = (b, g, r)
    name = "(b, g, r) = (%d, %d, %d)" % (b, g, r)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def info(_info):
    def decorator(_func):
        def parameters(*args, **kwargs):
            print("[info] {}".format(_info))
            exec_func = _func(*args, **kwargs)
            return exec_func

        return parameters

    return decorator


def showImage(*args):
    for _index, _arg in enumerate(args):
        cv2.imshow("img {}".format(_index), _arg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showImages(**kwargs):
    for _key in kwargs:
        cv2.imshow("{}".format(_key), kwargs[_key])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def splitChannel(_img):
    if _img.ndim == 2:
        return _img
    else:
        _bgr = [_img]

        for i in range(3):
            _temp = _img.copy()
            _temp[:, :, (i + 1) % 3] = 0
            _temp[:, :, (i + 2) % 3] = 0

            _bgr.append(_temp)

        return _bgr


def biBubic(_x):
    _x = abs(_x)
    if _x <= 1:
        return 1 - 2 * (_x ** 2) + (_x ** 3)
    elif _x < 2:
        return 4 - 8 * _x + 5 * (_x ** 2) - (_x ** 3)
    else:
        return 0


@info("biBubicInterpolation 內容似乎有瑕疵，需校正，請改用 biBubicInterpolation2(_img, _scale, _prefilter=True)")
def biBubicInterpolation(_img, _height_scale, _width_scale):
    # print("這個雙三次插值 (Bicubic interpolation)的內容似乎有瑕疵，需校正")
    if _img.ndim == 2:
        _height, _width = _img.shape
    else:
        _height, _width, _ = _img.shape

    _dst_height = int(_height * _height_scale)
    _dst_width = int(_width * _width_scale)
    _dst = np.zeros((_dst_height, _dst_width, 3), dtype=np.uint8)

    for _h in range(_dst_height):
        for _w in range(_dst_width):
            _x = _h * (_height / _dst_height)
            _y = _w * (_width / _dst_width)

            _x_index = math.floor(_x)
            _y_index = math.floor(_y)

            _u = _x - _x_index
            _v = _y - _y_index

            _temp = 0
            for _h_prime in [-1, 0, 1]:
                for _w_prime in [-1, 0, 1]:
                    if (_x_index + _h_prime < 0 or _y_index + _w_prime < 0 or
                            _x_index + _h_prime >= _height or _y_index + _w_prime >= _width):
                        continue
                    _temp += (_img[_x_index + _h_prime, _y_index + _w_prime] *
                              biBubic(_h_prime - _u) *
                              biBubic(_w_prime - _v))

            _dst[_h, _w] = np.clip(_temp, 0, 255)

    return _dst


def biBubicInterpolation2(_img, _scale, _prefilter=True):
    if _img.ndim == 2:
        _dst = scipy.ndimage.interpolation.zoom(_img, _scale, prefilter=_prefilter)

    else:
        b, g, r = cv2.split(_img)
        b = scipy.ndimage.interpolation.zoom(b, _scale, prefilter=_prefilter)
        g = scipy.ndimage.interpolation.zoom(g, _scale, prefilter=_prefilter)
        r = scipy.ndimage.interpolation.zoom(r, _scale, prefilter=_prefilter)

        _dst = cv2.merge([b, g, r])

    return _dst


# 最大公因數
def gcd(_a, _b):
    # https://www.geeksforgeeks.org/gcd-in-python/
    while _b > 0:
        _a, _b = _b, _a % _b

    return _a


# 最小公倍數
def lcm(_a, _b):
    # http://drweb.nksh.tp.edu.tw/student/lesson/G005/
    return _a * _b // gcd(_a, _b)


def showTrainHistory(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


def plotImage(image, _size_inches=2):
    fig = plt.gcf()
    fig.set_size_inches(_size_inches, _size_inches)
    plt.imshow(image, cmap='binary')
    plt.show()


def colorfulDataFrame(df, cmap=plt.cm.Blues):
    _df = df.copy()
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    #         繪圖數據   填充色       方塊的間隔     顯示數值
    sns.heatmap(_df, cmap=cmap, linewidths=0.1, annot=True)
    plt.show()


if __name__ == "__main__":
    showSingleColor(20, 40, 80)
    showSingleColor(40, 80, 160)
    showSingleColor(60, 120, 240)
