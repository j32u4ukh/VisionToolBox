from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from OpenEyes.utils import (
    showImage,
    showImages,
    showTrainHistory,
    plotImage,
    plotLabelsAndPrediction,
    colorfulDataFrame
)


np.random.seed(10)


# region Mnist model
def mnistDataPrepare():
    # region Prepare data
    (x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
    print("train data:", len(x_train_image))
    print(" test data:", len(x_test_image))

    # x_train_image.shape = (60000, 28, 28)
    print("x_train_image:", x_train_image.shape)
    # y_train_label: (60000,)
    print("y_train_label:", y_train_label.shape)
    plotImage(x_train_image[0])

    print("Train")
    plotLabelsAndPrediction(x_train_image, y_train_label, [], 0, 10)
    print("Test")
    plotLabelsAndPrediction(x_test_image, y_test_label, [], 0, 10)

    x_train = x_train_image.reshape(60000, 28 * 28).astype('float32')
    x_test = x_test_image.reshape(10000, 28 * 28).astype('float32')
    print("x_train:", x_train.shape)
    print("x_test:", x_test.shape)

    print("Image")
    print(x_train_image[0])
    print("Input")
    print(x_train[0])

    x_train_normalize = x_train / 255
    x_test_normalize = x_test / 255
    print(x_train_normalize[0])

    y_train_OneHot = np_utils.to_categorical(y_train_label)
    y_test_OneHot = np_utils.to_categorical(y_test_label)

    print("x_train_normalize.shape", x_train_normalize.shape)
    print("y_train_OneHot.shape", y_train_OneHot.shape)
    print("x_test_normalize.shape", x_test_normalize.shape)
    print("y_test_OneHot.shape", y_test_OneHot.shape)
    # endregion


def dense256():
    # region Build model
    model = Sequential()
    model.add(Dense(input_dim=28 * 28,
                    units=256,
                    activation="relu",
                    kernel_initializer='normal'))
    model.add(Dense(units=10,
                    activation="softmax",
                    kernel_initializer='normal'))
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    print(model.summary())

    # 開始訓練
    train_history = model.fit(x_train_normalize,
                              y_train_OneHot,
                              validation_split=0.2,
                              epochs=10,
                              batch_size=200,
                              verbose=2)
    scores = model.evaluate(x_test_normalize, y_test_OneHot)
    print("Accuray = ", scores[1])
    showTrainHistory(train_history, 'acc', 'val_acc')
    showTrainHistory(train_history, 'loss', 'val_loss')

    prediction = model.predict_classes(x_test_normalize)
    print(prediction)
    plotLabelsAndPrediction(x_test_image,
                            y_test_label,
                            prediction,
                            idx=340)

    # 混淆矩陣
    cross_tab = pd.crosstab(y_test_label,
                            prediction,
                            rownames=['label'],
                            colnames=['predict'])
    # print(cross_tab)
    colorfulDataFrame(df=cross_tab)

    df = pd.DataFrame({'label': y_test_label, 'predict': prediction})
    print(df.head())

    l7p2 = df[(df.label == 7) & (df.predict == 2)]
    print(l7p2)

    plotLabelsAndPrediction(x_test_image,
                            y_test_label,
                            prediction,
                            idx=9009,
                            num=1)
    # endregion


def dense1000():
    # Increase Neurons:256 to 1000
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=1000, activation="relu", kernel_initializer='normal'))
    model.add(Dense(units=10, activation="softmax", kernel_initializer='normal'))
    # 模型設定
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    print(model.summary())

    # 開始訓練
    train_history = model.fit(x_train_normalize, y_train_OneHot, validation_split=0.2, epochs=10, batch_size=200,
                              verbose=2)
    scores = model.evaluate(x_test_normalize, y_test_OneHot)
    print("Accuray = ", scores[1])

    showTrainHistory(train_history, 'acc', 'val_acc')
    showTrainHistory(train_history, 'loss', 'val_loss')


def dropoutDense1000():
    # 避免Overfitting 而加入Dropout layer
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=1000, activation="relu", kernel_initializer='normal'))

    # Add Dropout layer
    model.add(Dropout(0.5))

    model.add(Dense(units=10, activation="softmax", kernel_initializer='normal'))
    # 模型設定
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    print(model.summary())

    # 開始訓練
    train_history = model.fit(x_train_normalize, y_train_OneHot, validation_split=0.2, epochs=10, batch_size=200,
                              verbose=2)
    scores = model.evaluate(x_test_normalize, y_test_OneHot)
    print("Accuray = ", scores[1])

    showTrainHistory(train_history, 'acc', 'val_acc')
    showTrainHistory(train_history, 'loss', 'val_loss')


def dropoutDense1000_1000():
    # 2 layers with Dropout layer
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=1000, activation="relu", kernel_initializer='normal'))
    model.add(Dropout(0.5))

    # Second layer
    model.add(Dense(units=1000, activation="relu", kernel_initializer='normal'))
    model.add(Dropout(0.5))

    model.add(Dense(units=10, activation="softmax", kernel_initializer='normal'))
    # 模型設定
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    print(model.summary())

    # 開始訓練
    train_history = model.fit(x_train_normalize, y_train_OneHot, validation_split=0.2, epochs=10, batch_size=200,
                              verbose=2)
    scores = model.evaluate(x_test_normalize, y_test_OneHot)
    print("Accuray = ", scores[1])

    showTrainHistory(train_history, 'acc', 'val_acc')
    showTrainHistory(train_history, 'loss', 'val_loss')
# endregion


def xorModel():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="sgd")
    print(model.summary())

    model.fit(X, y, epochs=1000, batch_size=1, verbose=2)
    print(model.predict_proba(X))

    target = np.array([[0.1, 0.3]])
    print(model.predict_proba(target))
    print(model.predict(target))
    print(model.predict_classes(target))

    def xor_trainning():
        x_train = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])
        y_train = np.array([0, 1, 1, 0])
        xor_model = Sequential()
        #    , kernel_initializer = 'uniform'
        xor_model.add(Dense(input_dim=2, units=4, activation="tanh"))
        xor_model.add(Dropout(0.2))
        xor_model.add(Dense(units=1, activation="sigmoid"))
        # 模型設定
        xor_model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])
        train_history = xor_model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=2)

        fig = plt.gcf()
        fig.set_size_inches(15, 5)
        plt.subplot(1, 2, 1)
        plt.plot(train_history.history["acc"])
        plt.title('Train History')
        plt.ylabel("acc")
        plt.xlabel('Epoch')
        plt.legend(['acc'], loc='best')

        plt.subplot(1, 2, 2)
        plt.plot(train_history.history["loss"])
        plt.title('Train History')
        plt.ylabel("loss")
        plt.xlabel('Epoch')
        plt.legend(['loss'], loc='best')
        plt.show()
        return xor_model

    xor_model = xor_trainning()
    # score = xor_model.evaluate()

    target = np.array([[0.1, 0.3]])
    print(xor_model.predict_proba(target))
    print(xor_model.predict(target))
    print(xor_model.predict_classes(target))


mnistModel()
