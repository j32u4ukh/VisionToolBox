from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from utils import (
    showImage,
    showImages,
    showTrainHistory,
    plotImage,
    plotLabelsAndPrediction,
    colorfulDataFrame
)


# region Data prepare
np.random.seed(10)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 變成圖片格式 (28, 28, 1)
x_train4D = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_testn4D = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# data pre-processing
x_train4D_normalize = x_train4D / 255
x_testn4D_normalize = x_testn4D / 255
y_train_OneHot = np_utils.to_categorical(y_train)
y_test_OneHot = np_utils.to_categorical(y_test)
# endregion

# region Build model
model = Sequential()
model.add(Conv2D(filters=16,
                 kernel_size=(5, 5),
                 input_shape=(28, 28, 1),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))
# 一般来说，如果最后一层接上softmax作为分类概率输出时，都会用categorical_crossentropy作为损失函数
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
# endregion

train_history = model.fit(x_train4D_normalize,
                          y_train_OneHot,
                          validation_split=0.2,
                          epochs=10,
                          batch_size=300,
                          verbose=2)

# loss = 'categorical_crossentropy' metrics = ['accuracy']
loss_value, metrics_values = model.evaluate(x_testn4D_normalize, y_test_OneHot)
print("Accuray = ", metrics_values)
showTrainHistory(train_history)

prediction = model.predict_classes(x_testn4D_normalize)
print("Actual label:", y_test_OneHot)
print(prediction[:10])

plotLabelsAndPrediction(x_test, y_test, prediction, idx=0)

crosstab = pd.crosstab(y_test, prediction, rownames=['label'], colnames=['predict'])
colorfulDataFrame(crosstab)

print(x_train4D[0])
