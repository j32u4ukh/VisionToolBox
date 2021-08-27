from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import (
    Sequential,
    load_model
)
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from OpenEyes.utils import (
    showSingleColor,
    showImage,
    showImages,
    showTrainHistory,
    plotImage,
    plotLabelsAndPrediction,
    plotImagesLabelsPrediction,
    colorfulDataFrame
)


def showPredictedProbability(y, prediction, x_img, Predicted_Probability, i):
    print("label:", label_dict[y[i][0]], ", predict", label_dict[prediction[i]])
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(x_img[i], (32, 32, 3)))
    plt.show()
    for j in range(10):
        print(label_dict[j] + ":\t%1.9f" % Predicted_Probability[i][j])


np.random.seed(10)
(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()

# 維度 1:樣本數；維度 2, 3:影像大小(EX 32*32)；維度 4:RBG 三原色，所以是3
print("Train data: image: ", x_img_train.shape, " labels: ", y_label_train.shape)
print("Test data:  image: ", x_img_test.shape, " labels: ", y_label_test.shape)
print("x_img_test[0][0][0] is RBG color:", x_img_test[0][0][0])
r, b, g = x_img_test[8][0][0]
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

plotImagesLabelsPrediction(x_img_train, y_label_train, [], label_dict,  0, num=10)

# 將特徵值標準化，可提高模型準確度，並加速收斂速度
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
print("Normalized RGB color:", x_img_train_normalize[0][0][0])
print("y_label_train[:5]", y_label_train[:5])

y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
print("y_label_train_OneHot.shape", y_label_train_OneHot.shape)
print("y_label_train_OneHot[:5]", y_label_train_OneHot[:5])


def model1():
    # acc: 0.8478 - val_acc: 0.7361
    # 一個完整的卷積運算，包含1個卷積層和1個池化層
    model = Sequential()
    # 產生 filters = 32 個影像，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # input_shape = (32, 32, 3)：維度 1, 2:表示影像大小32X32；維度 3:RGB三原色，所以是3
    # 激活函數為 activation = 'relu'
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3),
                     activation='relu', padding='same'))
    # 加入 Dropout 避免 overfitting
    # Dropout(rate = 0.25)：每次訓練迭代時，隨機放棄25%的神經元，以避免 overfitting
    # 正確來說是，每次只訓練(1-0.25)，即 75%的神經元
    model.add(Dropout(rate=0.25))
    # pool_size = (2, 2)執行縮減取樣，將32X32的影像大小縮小為16X16，但影像數量不變仍為32
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 建立卷積層2和池化層2
    # 產生 filters = 64 個影像(原本32個)，濾鏡大小為3X3，padding = 'same'使卷積運算後產生影像大小不變
    # input_shape 輸入層才需要；激活函數為 activation = 'relu'
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    # 加入 Dropout 避免 overfitting
    model.add(Dropout(rate=0.25))
    # pool_size = (2, 2)執行縮減取樣，將16X16的影像大小縮小為8X8，但影像數量不變仍為64
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 建立平坦層：將前面 64個 8X8 的影像轉為一維資料。長度為64X8X8=4096(個 float 數字 / 神經元)
    model.add(Flatten())
    model.add(Dropout(rate=0.25))

    # 建立隱藏層，1024個神經元
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=10, activation='softmax'))
    # loss = 'categorical_crossentropy' 損失函數，通常使用cross entropy 交叉熵
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # %%
    # 開始訓練
    # validation_split = 0.2，20% 驗證資料 80% 訓練資料
    # epochs = 10 訓練週期為10；batch_size = 128 每一批次訓練 128筆資料
    train_history = model.fit(x_img_train_normalize, y_label_train_OneHot, validation_split=0.2,
                              epochs=10, batch_size=128, verbose=1)
    # showTrainHistory(train_history)
    # %% 模型評估
    loss_value, metrics_value = model.evaluate(x_img_test_normalize, y_label_test_OneHot, verbose=0)
    print("loss_value:", loss_value, ", metrics_value:", metrics_value)
    # %%
    # 進行預測
    prediction = model.predict_classes(x_img_test_normalize)
    print("y_label_test[:10]", y_label_test[:10])
    print("prediction[:10]", prediction[:10])
    # %%
    plotLabelsAndPrediction(x_img_test, y_label_test, prediction, 0, 10)
    # %%
    Predicted_Probability = model.predict(x_img_test_normalize)
    # Predicted_Probability.shape = (10000, 10)
    # %%
    print(Predicted_Probability[0])

    # 正確預測為 貓
    showPredictedProbability(y_label_test, prediction, x_img_test, Predicted_Probability, 0)
    # 錯誤預測，將 飛機 預測為 船
    showPredictedProbability(y_label_test, prediction, x_img_test, Predicted_Probability, 3)

    # 混淆矩陣(confusion matrix)，又稱誤差矩陣(error matrix)
    # 用來得知，我們的模型中，哪個類別的預測準確率最高
    # 以特定表格形式呈現，得以視覺化
    # 查看 prediction 預測結果的形狀(結果為1維資料)
    print("prediction.shape", prediction.shape)
    # 查看 y_label_test 實際的形狀(結果為2維資料)
    print("y_label_test.shape", y_label_test.shape)
    # %%
    # 將 y_label_test 轉為1維資料
    y_label_test = y_label_test.reshape(-1)
    print("y_label_test.shape", y_label_test.shape)
    # %%
    print(label_dict)
    pd.crosstab(y_label_test, prediction, rownames=["label"], colnames=["predict"])


def model2():
    # %% Top acc: 0.8505 - val_acc: 0.8023
    dropout_rate = 0.35
    # Deeper CNN network
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout_rate))

    model.add(Flatten())
    model.add(Dense(units=2500, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=250, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # start trainning
    train_history = model.fit(x_img_train_normalize, y_label_train_OneHot, validation_split=0.2,
                              epochs=50, batch_size=500, verbose=2)
    # showTrainHistory(train_history)
    # %%
    loss_value, metrics_value = model.evaluate(x_img_test_normalize, y_label_test_OneHot, verbose=1)
    print("Accuray = ", metrics_value)
    # %%
    model.save_weights("./SaveModel/cifarCnnModel.h5")
    print("Save model")


# %% 載入模型


# 刪除既有模型變數
if locals().__contains__("model"):
    del model

# 載入模型
model = load_model('./SaveModel/cifarCnnModel.h5')
