from time import time

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from matplotlib import pyplot as plt
import numpy as np

from OpenEyes.utils import (
    plotLabelsAndPrediction,
    plotImage,
    showImage,
    showImages
)


def plot_image(image):
    img = image.reshape(28, 28)
    plt.imshow(img, cmap='binary')
    plt.show()


def layer(input_dim, output_dim, inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs


def plotResult(x, y, label):
    fig = plt.gcf()
    fig.set_size_inches(4, 2)
    plt.plot(x, y, label=label)
    plt.xlabel("epoch")
    plt.ylabel(label)
    plt.legend([label], loc="best")
    plt.show()


mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
print("train images:", mnist.train.images.shape)  # (55000, 784)
print("train labels:", mnist.train.labels.shape)  # (55000, 10)

print("validation images:", mnist.validation.images.shape)  # (5000, 784)
print("validation labels:", mnist.validation.labels.shape)  # (5000, 10)

print("test images:", mnist.test.images.shape)  # (10000, 784)
print("test labels:", mnist.test.labels.shape)  # (10000, 10)

img = mnist.train.images[0]
img = img.reshape(28, 28)
print("img.shape:", img.shape)
showImage(img)
plotImage(img, _size_inches=3)

print(np.argmax(mnist.train.labels[0]))
print(mnist.train.labels[0])

plotLabelsAndPrediction(mnist.train.images, mnist.train.labels, [], 0)
plotLabelsAndPrediction(mnist.test.images, mnist.test.labels, [], 0)

batch_images_xs, batch_labels_ys = mnist.train.next_batch(batch_size=100)
plotLabelsAndPrediction(batch_images_xs, batch_labels_ys, [], 0)


# region Build model
# placeholder
x = tf.placeholder(tf.float32, [None, 784])
y_label = tf.placeholder(tf.float32, [None, 10])

h1 = layer(784, 256, x, tf.nn.relu)
y_predict = layer(256, 10, h1)

# loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict,
                                                           labels=y_label)
loss_function = tf.reduce_mean(cross_entropy)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_function)

# Evaluate Accuracy
correct_prediction = tf.equal(tf.argmax(y_label, 1),
                              tf.argmax(y_predict, 1))
# correct_prediction 利用 tf.cast 轉型為 tf.float32
# 再利用 tf.reduce_mean 將所有數值平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# endregion

trainEpochs = 15
batchSize = 100
totalBatchs = int(mnist.train.num_examples / batchSize)
epoch_list = []
loss_list = []
accuracy_list = []
startTime = time()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        sess.run(optimizer, feed_dict={x: batch_x, y_label: batch_y})

    loss, acc = sess.run([loss_function, accuracy],
                         feed_dict={x: mnist.validation.images,
                                    y_label: mnist.validation.labels})
    epoch_list.append(epoch)
    loss_list.append(loss)
    accuracy_list.append(acc)
    print("Epoch: %02d" % (epoch + 1),
          "Loss: ", "{:.5f}".format(loss), "Accuracy: ", acc)
duration = time() - startTime
print("Train cost time: ", duration)

plotResult(epoch_list, loss_list, label="loss")
plotResult(epoch_list, accuracy_list, label="accuracy")

# Evaluate Accuracy
print("Accuracy:", sess.run(accuracy,
                            feed_dict={x: mnist.test.images,
                                       y_label: mnist.test.labels}))

sess.run(tf.global_variables_initializer())
prediction_result = sess.run(tf.argmax(y_predict, 1),
                             feed_dict={x: mnist.test.images})

print(prediction_result[:10])
plotLabelsAndPrediction(mnist.test.images,
                        mnist.test.labels,
                        prediction_result, 0)

sess.close()

# Neurons 256>>1000
x = tf.placeholder(tf.float32, [None, 784])
h1 = layer(784, 1000, x, tf.nn.relu)
y_predict = layer(1000, 10, h1)
y_label = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict,
                                                           labels=y_label)
loss_function = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)
correct_prediction = tf.equal(tf.argmax(y_label, 1),
                              tf.argmax(y_predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

trainEpochs = 15
batchSize = 100
totalBatchs = int(mnist.train.num_examples / batchSize)
epoch_list = []
loss_list = []
accuracy_list = []

startTime = time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(trainEpochs):
        for i in range(totalBatchs):
            batch_x, batch_y = mnist.train.next_batch(batchSize)
            sess.run(optimizer, feed_dict={x: batch_x, y_label: batch_y})

        loss, acc = sess.run([loss_function, accuracy],
                             feed_dict={x: mnist.validation.images,
                                        y_label: mnist.validation.labels})
        epoch_list.append(epoch)
        loss_list.append(loss)
        accuracy_list.append(acc)
        print("Epoch: %02d" % (epoch + 1),
              "Loss: ", "{:.5f}".format(loss), "Accuracy: ", acc)
    duration = time() - startTime
    print("Train cost time: ", duration)
    print("Accuracy:", sess.run(accuracy,
                                feed_dict={x: mnist.test.images,
                                           y_label: mnist.test.labels}))
    prediction_result = sess.run(tf.argmax(y_predict, 1),
                                 feed_dict={x: mnist.test.images})

plotLabelsAndPrediction(mnist.test.images,
                        mnist.test.labels,
                        prediction_result, 0)

# one hidden layer >> two hidden layer
x = tf.placeholder(tf.float32, [None, 784])
h1 = layer(784, 1000, x, tf.nn.relu)
h2 = layer(1000, 1000, h1, tf.nn.relu)
y_predict = layer(1000, 10, h2)
y_label = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict,
                                                           labels=y_label)
loss_function = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)
correct_prediction = tf.equal(tf.argmax(y_label, 1),
                              tf.argmax(y_predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

trainEpochs = 15
batchSize = 100
totalBatchs = int(mnist.train.num_examples / batchSize)
epoch_list = []
loss_list = []
accuracy_list = []

startTime = time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(trainEpochs):
        for i in range(totalBatchs):
            batch_x, batch_y = mnist.train.next_batch(batchSize)
            sess.run(optimizer, feed_dict={x: batch_x, y_label: batch_y})

        loss, acc = sess.run([loss_function, accuracy],
                             feed_dict={x: mnist.validation.images,
                                        y_label: mnist.validation.labels})
        epoch_list.append(epoch)
        loss_list.append(loss)
        accuracy_list.append(acc)
        print("Epoch: %02d" % (epoch + 1),
              "Loss: ", "{:.5f}".format(loss), "Accuracy: ", acc)
    duration = time() - startTime
    print("Train cost time: ", duration)
    print("Accuracy:", sess.run(accuracy,
                                feed_dict={x: mnist.test.images,
                                           y_label: mnist.test.labels}))
    prediction_result = sess.run(tf.argmax(y_predict, 1),
                                 feed_dict={x: mnist.test.images})
plotLabelsAndPrediction(mnist.test.images,
                        mnist.test.labels,
                        prediction_result, 0)


# region CNN by tensorflow
# %% 1.模型預處理
mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="W")


def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name="b")


def conv2d(x, W):
    # padding="SAME" 使輸入輸出圖像大小相同
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    """參數有四個，和卷積很類似：

    第一個參數value：需要池化的輸入，一般池化層接在卷積層後面，所以輸入通常是feature map，
    依然是[batch, height, width, channels]這樣的shape

    第二個參數ksize：池化窗口的大小，取一個四維向量，一般是[1, height, width, 1]，因為我們不想在batch和channels上做池化，
    所以這兩個維度設為了1

    第三個參數strides：和卷積類似，窗口在每一個維度上滑動的步長，一般也是[1, stride,stride,1]

    第四個參數padding：和卷積類似，可以取'VALID' 或者'SAME'"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding="SAME")


# %% 2.建立模型
with tf.name_scope("Input_Layer"):
    x = tf.placeholder("float", shape=[None, 784], name="x")
    # 因後續輸入筆數不固定，因此第 1 維設 -1
    # 第 2, 3維 影像大小為 28*28
    # 第 4維 灰階圖片僅需1個維度，若為彩色則須設為 3
    # x_image shape=(?, 28, 28, 1)
    x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope("C1_Conv"):
    # 第 1, 2維 代表濾鏡大小為 5*5
    # 第 3維 灰階圖片僅需1個維度，若為彩色則須設為 3
    # 第 4維 產生 16個影像
    W1 = weight([5, 5, 1, 16])
    b1 = bias([16])
    # Conv1 shape=(?, 28, 28, 16) dtype=float32>
    Conv1 = conv2d(x_image, W1) + b1
    # C1_Conv shape=(?, 28, 28, 16)
    C1_Conv = tf.nn.relu(Conv1)

with tf.name_scope("C1_Pool"):
    # C1_Pool shape=(?, 14, 14, 16)
    C1_Pool = max_pool_2x2(C1_Conv)

with tf.name_scope("C2_Conv"):
    # 16個灰階圖片，因此第3維設16
    W2 = weight([5, 5, 16, 36])
    b2 = bias([36])
    # Conv2 shape=(?, 14, 14, 36)
    Conv2 = conv2d(C1_Pool, W2) + b2
    # C2_Conv shape=(?, 14, 14, 36)
    C2_Conv = tf.nn.relu(Conv2)

with tf.name_scope("C2_Pool"):
    # C2_Pool shape=(?, 7, 7, 36)
    C2_Pool = max_pool_2x2(C2_Conv)

with tf.name_scope("D_Flat"):
    # D_Flat shape=(?, 1764) for 卷積層 接 全連接層
    D_Flat = tf.reshape(C2_Pool, [-1, 1764])

with tf.name_scope("D_Hidden_Layer"):
    W3 = weight([1764, 128])
    b3 = bias([128])
    # D_Hidden shape=(?, 128)
    D_Hidden = tf.nn.relu(tf.matmul(D_Flat, W3) + b3)

    # D_Hidden_Dropout shape=(?, 128)
    # rate=0.2 隨機去掉 20% 神經元，保留 80% 神經元
    D_Hidden_Dropout = tf.nn.dropout(D_Hidden, rate=0.2)

with tf.name_scope("Output_Layer"):
    W4 = weight([128, 10])
    b4 = bias([10])
    # y_predict shape=(?, 10)
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden_Dropout, W4) + b4)

# 3.定義訓練方式
with tf.name_scope("optimizer"):
    y_label = tf.placeholder("float", shape=[None, 10], name="y_label")

    # cross_entropy shape=(?,)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict,
                                                               labels=y_label)
    loss_function = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_function)

# 4.評估模型好壞
with tf.name_scope("evaluate_model"):
    # correct_prediction shape=(?,) dimension=1 按行找
    correct_prediction = tf.equal(tf.argmax(y_predict, 1),
                                  tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 5.進行訓練
trainEpochs = 30
batchSize = 100
totalBatchs = int(mnist.train.num_examples / batchSize)
epoch_list = []
loss_list = []
accuracy_list = []

startTime = time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        sess.run(optimizer, feed_dict={x: batch_x, y_label: batch_y})

    loss, acc = sess.run([loss_function, accuracy],
                         feed_dict={x: mnist.validation.images,
                                    y_label: mnist.validation.labels})
    epoch_list.append(epoch)
    loss_list.append(loss)
    accuracy_list.append(acc)
    # https://www.runoob.com/python/att-string-format.html
    print("Epoch: {:0>2d}, Loss: {:.4f}, Accuracy: {:.4f}".format((epoch + 1), loss, acc))
duration = time() - startTime
print("Train cost time: ", duration)

plotResult(epoch_list, loss_list, "loss")
plotResult(epoch_list, accuracy_list, "accuracy")
print("Accuracy:", sess.run(accuracy,
                            feed_dict={x: mnist.test.images,
                                       y_label: mnist.test.labels}))
prediction_result = sess.run(tf.argmax(y_predict, 1),
                             feed_dict={x: mnist.test.images})
plotLabelsAndPrediction(mnist.test.images,
                        mnist.test.labels,
                        prediction_result, 0)

"""特徵圖 strat"""
feed_dict = {x: mnist.test.images[:1], y_label: mnist.test.labels[:1]}
conv1 = sess.run(Conv1, feed_dict=feed_dict)
conv1_relu = sess.run(C1_Conv, feed_dict=feed_dict)
conv2_relu = sess.run(C2_Conv, feed_dict=feed_dict)

f_conv1_relu = conv1_relu[0, :, :, 6]
showImage(f_conv1_relu)

f_conv2_relu = conv2_relu[0, :, :, 6]
showImage(f_conv2_relu)


"""特徵圖 end"""
# region For tensor board
# https://blog.csdn.net/u012436149/article/details/53184847

# merged = tf.summary.merge_all()
# teain_writer = tf.summary.FileWriter("log/CH17-CNN", sess.graph)

# endregion

sess.close()
# endregion
