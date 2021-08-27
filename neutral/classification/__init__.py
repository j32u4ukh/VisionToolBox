import cv2
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D
)
from keras.models import Sequential
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from utils import (
    plotImagesLabelsPrediction,
    showSingleColor,
    showTrainHistory
)


# region Cats and dogs classification
def catsAndDogsDataPrepare(is_grayscale, is_normalized):
    cifar10 = Cifar10()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data([3, 5])
    print("Train data: image: ", x_train.shape, " labels: ", y_train.shape)
    print("Test data:  image: ", x_test.shape, " labels: ", y_test.shape)
    r, b, g = x_train[0][0][0]
    showSingleColor(b, g, r)

    label_dict = {0: "airplae",
                  1: "automobile",
                  2: "bird",
                  3: "cat",
                  4: "deer",
                  5: "dog",
                  6: "frog",
                  7: "horse",
                  8: "ship",
                  9: "truck"}

    plotImagesLabelsPrediction(x_train, y_train, [], label_dict, 0)

    # 修改標籤，從 0 開始
    syu_ru_i = [3, 5]
    for sri in range(len(syu_ru_i)):
        print("sri", sri)
        number = syu_ru_i[sri]
        indexs = np.where(y_train[:, 0] == number)[0]
        for i in indexs:
            y_train[i][0] = sri

        indexs = np.where(y_test[:, 0] == number)[0]
        for i in indexs:
            y_test[i][0] = sri
    print("y_train")
    print(y_train[:5])
    print("y_test")
    print(y_test[:5])

    x_train_normal = x_train / 255.0
    x_test_normal = x_test / 255.0
    y_train_onehot = np_utils.to_categorical(y_train)
    y_test_onehot = np_utils.to_categorical(y_test)

    print("Train data: image: ", x_train.shape, " labels: ", y_train.shape)
    print("Test data:  image: ", x_test.shape, " labels: ", y_test.shape)

    print("x_train_normal[0][0][:5]")
    print(x_train_normal[0][0][:5])
    print("y_train_onehot[:5]")
    print(y_train_onehot[:5])

    # =======================================================================

    cifar10 = Cifar10()
    x, y = cifar10.load_data([3, 5])
    x, y = cifar10.shuffleData(x, y)
    (x_train, y_train), (x_test, y_test) = cifar10.splitTrainTest(x, y)
    # %% 修改標籤，從 0 開始
    syu_ru_i = [3, 5]
    for sri in range(len(syu_ru_i)):
        print("sri", sri)
        number = syu_ru_i[sri]
        indexs = np.where(y_train[:, 0] == number)[0]
        for i in indexs:
            y_train[i][0] = sri

        indexs = np.where(y_test[:, 0] == number)[0]
        for i in indexs:
            y_test[i][0] = sri

    label_dict = {0: "airplae",
                  1: "automobile",
                  2: "bird",
                  3: "cat",
                  4: "deer",
                  5: "dog",
                  6: "frog",
                  7: "horse",
                  8: "ship",
                  9: "truck"}

    plotImagesLabelsPrediction(x_train, y_train, [], label_dict, 0)

    # 彩圖轉灰階 (9600, 32, 32, 3) >> (9600, 32, 32)
    gray_train = None
    for img in x_train:
        gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)])
        if gray_train is None:
            gray_train = gray
        else:
            gray_train = np.concatenate((gray_train, gray), axis=0)
    print("gray_train.shape", gray_train.shape)

    img0 = x_train[0]
    img1 = x_train[1]
    gray0 = np.array([cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)])
    gray1 = np.array([cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)])
    gray = np.concatenate((gray0, gray1), axis=0)
    print(gray.shape)

    # 彩圖轉灰階 (2400, 32, 32, 3) >> (2400, 32, 32)
    gray_test = None
    for img in x_test:
        gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)])
        if gray_test is None:
            gray_test = gray
        else:
            gray_test = np.concatenate((gray_test, gray), axis=0)
    print("gray_test.shape", gray_test.shape)

    x_train_normal = gray_train / 255.0
    x_test_normal = gray_test / 255.0
    y_train_onehot = np_utils.to_categorical(y_train)
    y_test_onehot = np_utils.to_categorical(y_test)

    print("x_train_normal.shape", x_train_normal.shape, "x_test_normal.shape", x_test_normal.shape)
    x_train_normal = x_train_normal.reshape(-1, 32, 32, 1)
    x_test_normal = x_test_normal.reshape(-1, 32, 32, 1)
    print("x_train_normal.shape", x_train_normal.shape, "x_test_normal.shape", x_test_normal.shape)


def colorCatsAndDogs1():
    # (彩色)貓狗辨識 1
    # 建立模型
    dropout_rate = 0.4
    model = Sequential()

    # 卷積層 1
    # 產生 filters = 32 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 32X32X32 = 32768
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     input_shape=(32, 32, 3),
                     activation='relu',
                     padding='same'))
    # pool_size = (2, 2)執行縮減取樣，將32X32的影像大小縮小為16X16，但影像數量不變仍為 32
    # 數量為: 16X16X32 = 8192
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 2
    # 產生 filters = 64 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 16X16X64 = 16384
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    # pool_size = (2, 2)執行縮減取樣，將 16X16 的影像大小縮小為 8X8 ，但影像數量不變仍為 64
    # 數量為: 8X8X64 = 4096
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 3
    # 產生 filters = 128 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 8X8X128 = 8192
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    # pool_size = (2, 2)執行縮減取樣，將 8X8 的影像大小縮小為 4X4 ，但影像數量不變仍為 128
    # 數量為: 4X4X128 = 2048
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    # 建立平坦層：將前面 128 個 4X4 的影像轉為一維資料。長度為 4X4X128 = 2048 (個 float 數字 / 神經元)
    model.add(Flatten())
    model.add(Dropout(rate=dropout_rate))

    # 建立隱藏層， 1024 個神經元
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    # 建立隱藏層， 512 個神經元
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    # 輸出層(貓或狗)
    model.add(Dense(units=2, activation='sigmoid'))
    # loss = 'categorical_crossentropy' 損失函數，通常使用cross entropy 交叉熵
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # %% acc: 0.8081 - val_acc: 0.7833
    train_history = model.fit(x_train_normal,
                              y_train_onehot,
                              validation_split=0.2,
                              epochs=50,
                              batch_size=100,
                              verbose=2)
    showTrainHistory(train_history)


def colorCatsAndDogs2():
    # %% (彩色)貓狗辨識 2
    # 建立模型
    dropout_rate = 0.4
    model = Sequential()

    # 卷積層 1
    # 產生 filters = 32 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 32X32X32 = 32768
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     input_shape=(32, 32, 3),
                     activation='relu',
                     padding='same'))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 2
    # 產生 filters = 32 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 32X32X32 = 32768
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     input_shape=(32, 32, 3),
                     activation='relu',
                     padding='same'))
    # pool_size = (2, 2)執行縮減取樣，將32X32的影像大小縮小為16X16，但影像數量不變仍為 32
    # 數量為: 16X16X32 = 8192
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 3
    # 產生 filters = 64 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 16X16X64 = 16384
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 4
    # 產生 filters = 64 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 16X16X64 = 16384
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    # pool_size = (2, 2)執行縮減取樣，將 16X16 的影像大小縮小為 8X8 ，但影像數量不變仍為 64
    # 數量為: 8X8X64 = 4096
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 5
    # 產生 filters = 128 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 8X8X128 = 8192
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 6
    # 產生 filters = 128 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 8X8X128 = 8192
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    # pool_size = (2, 2)執行縮減取樣，將 8X8 的影像大小縮小為 4X4 ，但影像數量不變仍為 128
    # 數量為: 4X4X128 = 2048
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    # 建立平坦層：將前面 128 個 4X4 的影像轉為一維資料。長度為 4X4X128 = 2048 (個 float 數字 / 神經元)
    model.add(Flatten())
    model.add(Dropout(rate=dropout_rate))

    # 建立隱藏層， 1024 個神經元
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    # 建立隱藏層， 512 個神經元
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    # 輸出層(貓或狗)
    model.add(Dense(units=2, activation='sigmoid'))
    # loss = 'categorical_crossentropy' 損失函數，通常使用cross entropy 交叉熵
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    # %% acc: 0.8079 - val_acc: 0.8029
    train_history = model.fit(x_train_normal,
                              y_train_onehot,
                              validation_split=0.2,
                              epochs=50,
                              batch_size=200,
                              verbose=2)
    showTrainHistory(train_history)


def grayCatsAndDogs1():
    # (灰階)貓狗辨識 1
    # 建立模型
    dropout_rate = 0.4
    model = Sequential()

    # 卷積層 1
    # 產生 filters = 32 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 32X32X32 = 32768
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     input_shape=(32, 32, 1),
                     activation='relu',
                     padding='same'))
    # pool_size = (2, 2)執行縮減取樣，將32X32的影像大小縮小為16X16，但影像數量不變仍為 32
    # 數量為: 16X16X32 = 8192
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 2
    # 產生 filters = 64 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 16X16X64 = 16384
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    # pool_size = (2, 2)執行縮減取樣，將 16X16 的影像大小縮小為 8X8 ，但影像數量不變仍為 64
    # 數量為: 8X8X64 = 4096
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 3
    # 產生 filters = 128 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 8X8X128 = 8192
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    # pool_size = (2, 2)執行縮減取樣，將 8X8 的影像大小縮小為 4X4 ，但影像數量不變仍為 128
    # 數量為: 4X4X128 = 2048
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    # 建立平坦層：將前面 128 個 4X4 的影像轉為一維資料。長度為 4X4X128 = 2048 (個 float 數字 / 神經元)
    model.add(Flatten())
    model.add(Dropout(rate=dropout_rate))

    # 建立隱藏層， 512 個神經元
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    # 建立隱藏層， 128 個神經元
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    # 建立隱藏層， 32 個神經元
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    # 輸出層(貓或狗)
    model.add(Dense(units=2, activation='sigmoid'))
    # loss = 'categorical_crossentropy' 損失函數，通常使用cross entropy 交叉熵
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # %% acc: 0.8583 - val_acc: 0.7974 (目前 overfitting 嚴重)
    train_history = model.fit(x_train_normal,
                              y_train_onehot,
                              validation_split=0.2,
                              epochs=60,
                              batch_size=200,
                              verbose=2)
    showTrainHistory(train_history)


def grayCatsAndDogs2():
    # %% (灰階)貓狗辨識 2
    # 建立模型(加深網路, 樣本數少，容易 overfitting)
    dropout_rate = 0.4
    model = Sequential()

    # 卷積層 1
    # 產生 filters = 32 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 32X32X32 = 32768
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     input_shape=(32, 32, 1),
                     activation='relu',
                     padding='same'))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 2
    # 產生 filters = 32 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 32X32X32 = 32768
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    # pool_size = (2, 2)執行縮減取樣，將32X32的影像大小縮小為16X16，但影像數量不變仍為 32
    # 數量為: 16X16X32 = 8192
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 3
    # 產生 filters = 64 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 16X16X64 = 16384
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 4
    # 產生 filters = 64 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 16X16X64 = 16384
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    # pool_size = (2, 2)執行縮減取樣，將 16X16 的影像大小縮小為 8X8 ，但影像數量不變仍為 64
    # 數量為: 8X8X64 = 4096
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 5
    # 產生 filters = 128 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 8X8X128 = 8192
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    model.add(Dropout(rate=dropout_rate))

    # 卷積層 6
    # 產生 filters = 128 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # 數量為: 8X8X128 = 8192
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    # pool_size = (2, 2)執行縮減取樣，將 8X8 的影像大小縮小為 4X4 ，但影像數量不變仍為 128
    # 數量為: 4X4X128 = 2048
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    # 建立平坦層：將前面 128 個 4X4 的影像轉為一維資料。長度為 4X4X128 = 2048 (個 float 數字 / 神經元)
    model.add(Flatten())
    model.add(Dropout(rate=dropout_rate))

    # 建立隱藏層， 1024 個神經元
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    # 建立隱藏層， 512 個神經元
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    # 輸出層(貓或狗)
    model.add(Dense(units=2, activation='sigmoid'))
    # loss = 'categorical_crossentropy' 損失函數，通常使用cross entropy 交叉熵
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # %% acc: 0.5040 - val_acc: 0.4932 應該是 overfitting 了
    train_history = model.fit(x_train_normal,
                              y_train_onehot,
                              validation_split=0.2,
                              epochs=50,
                              batch_size=100,
                              verbose=2)
    showTrainHistory(train_history)
# endregion

