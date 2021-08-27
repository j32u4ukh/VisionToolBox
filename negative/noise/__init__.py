import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

from neutral.utils import showImage


# region 添加噪音
# 椒鹽噪音（salt-and-pepper noise）
def addNoise1(_img, _number):
    # http://www.voidcn.com/article/p-fdujibnd-bu.html
    # 椒鹽噪音（salt-and-pepper noise）
    _noise = _img.copy()

    # 產生多少噪音點
    for _num in range(_number):
        # 隨機座標
        x = int(np.random.random() * _noise.shape[0])
        y = int(np.random.random() * _noise.shape[1])

        # 如果是灰度图
        if _noise.ndim == 2:
            _noise[x, y] = 255
        # 如果是RBG图片
        elif _noise.ndim == 3:
            _noise[x, y] = (255, 255, 255)

    return _noise


def addNoise1Test():
    img = cv2.imread("data/Lenna.jpg")
    saltImage = addNoise1(img, 3000)
    showImage(img, saltImage)


# region https://jinzhangyu.github.io/2018/09/03/2018-09-03-
# OpenCV-Python%E6%95%99%E7%A8%8B-13-%E5%B9%B3%E6%BB%91%E5%9B%BE%E5%83%8F/
# 椒鹽噪聲
def addNoise2():
    # 加载图像并显示
    img = cv2.imread("data/Lenna.jpg", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 加上椒鹽噪聲
    dst1 = skimage.util.random_noise(image=img, mode='salt', clip=True, amount=0.2)
    dst2 = skimage.util.random_noise(image=img, mode='pepper', clip=True, amount=0.2)
    dst3 = skimage.util.random_noise(image=img, mode='s&p', clip=True, amount=0.2, salt_vs_pepper=0.5)

    showImage(dst1, dst2, dst3)
    """可以看到skimage.util.random_noise()是对每个通道都加噪的，这就导致了拼起来的图片的椒盐噪点并不是黑白的，而是三色的。"""

    # 绘制图片
    images = [[img], [dst1], [dst2], [dst3]]
    titles = ['Original', 'Salt', 'Pepper', 'Salt & Pepper']
    channel = ['', ' Reds', ' Greens', ' Blues']

    # 计算各个通道的值
    for i in range(4):
        # 4 種噪音後面，再將三個顏色通道區分出來。
        for j in range(3):
            temp = images[i][0].copy()
            temp[:, :, (j + 1) % 3] = 0
            temp[:, :, (j + 2) % 3] = 0
            # 將三個通道的圖案加入
            images[i].append(temp)

    # 绘制原图
    plt.figure(figsize=(14, 9))

    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, 4 * i + j + 1)
            plt.imshow(images[i][j])
            plt.title(titles[i] + channel[j], fontsize=10)
            plt.xticks([])
            plt.yticks([])

    plt.show()


# 高斯噪音
def addNoise3():
    # 加载图像并显示
    img = cv2.imread("data/Lenna.jpg", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 加上高斯噪音
    """不用Box-Muller方法，我们选择现成的skimage.util.random_noise()函数"""
    dst = skimage.util.random_noise(image=img,
                                    mode='gaussian',
                                    clip=True,
                                    mean=0.,
                                    var=0.01)

    showImage(dst)

    # 绘制图片
    images = [[img], [dst]]
    titles = ['Original', 'Gaussian']
    channel = ['', ' Reds', ' Blues', ' Greens']

    # 计算各个通道的值
    for i in range(len(images)):
        for j in range(3):
            temp = images[i][0].copy()
            temp[:, :, (j + 1) % 3] = 0
            temp[:, :, (j + 2) % 3] = 0
            images[i].append(temp)

    # 绘制原图
    plt.figure(figsize=(10, 8))

    for i in range(len(images)):
        for j in range(4):
            plt.subplot(len(images), 4, 4 * i + j + 1), plt.imshow(images[i][j])
            plt.title(titles[i] + channel[j], fontsize=10), plt.xticks([]), plt.yticks([])

    plt.show()
    """與上一節椒鹽噪聲相比，經過高斯噪聲的圖像，畫面不清晰，畫質很不好。
        這是因為，椒鹽噪聲是 出現在隨機位置、噪點深度基本固定 的噪聲；
        高斯噪聲與其相反，是 幾乎每個點上都出現噪聲、噪點深度隨機 的噪聲。"""
# endregion


# region https://www.jb51.net/article/162073.htm
# 椒鹽噪聲
def addNoise4(_img, _pepper):
    """添加椒鹽噪聲
    _pepper:黑噪聲比例 """
    output = np.zeros(_img.shape, np.uint8)
    rows, cols, _ = _img.shape
    _salt = 1 - _pepper

    for x in range(rows):
        for y in range(cols):
            rand = random.random()

            if rand > _pepper:
                if rand > _salt:
                    output[x][y] = 255
                else:
                    output[x][y] = _img[x][y]

    return output


def addNoise4Test():
    img = cv2.imread("data/Lenna.jpg")
    output = addNoise4(img, _pepper=0.05)
    showImage(output)


# 高斯噪聲
def addNoise5(_img, _mean=0, _var=0.001):
    """添加高斯噪聲
    _mean : 均值
    _var : 方差"""
    _image = _img.copy()
    _image = np.array(_image / 255, dtype=float)
    _noise = np.random.normal(_mean, _var ** 0.5, _img.shape)
    _result = _image + _noise
    # np.clip 將第一個數組，小於下限值者，變成下限值；大於上限值者，變成上限值
    # _result 範圍變成 0 ~ 1
    _result = np.clip(_result, 0, 1)
    _result = np.uint8(_result * 255)

    return _result


def addNoise5Test():
    img = cv2.imread("data/Lenna.jpg")
    output = addNoise5(img, _mean=0, _var=0.05)
    showImage(output)
# endregion


# region https://feelncut.com/2018/09/11/182.html
# 霧霾效果
def addNoise6(image, alpha=0.6, light=1):
    """添加霧霾
    alpha : 透視率 0~1
    light : 大氣光照"""
    img = np.array(image / 255) * alpha + light * (1 - alpha)
    img = np.clip(img, 0, 1)
    out = np.uint8(img * 255)
    return out


def addNoise6Test():
    img = cv2.imread("data/Lenna.jpg")
    output = addNoise6(img, 0.6, 1)
    showImage(output)


# 對比度與亮度
def addNoise7(_img, _contrast=1, _bright=0):
    """調整對比度與亮度(效果不佳，當 _contrast 或 _bright 較大時)
    _contrast : 對比度，調節對比度應該與亮度同時調節
    _bright : 亮度"""
    _image = _contrast * _img.copy() + _bright
    _image = np.uint8(np.clip(_image, 0, 255))

    return _image


def addNoise7Test():
    img = cv2.imread("data/Lenna.jpg")
    out1 = addNoise7(img, _contrast=1, _bright=0)
    out2 = addNoise7(img, _contrast=5, _bright=0)
    out3 = addNoise7(img, _contrast=1, _bright=5)
    out4 = addNoise7(img, _contrast=1, _bright=10)
    showImage(img, out1, out2, out3, out4)


# HSV
def addNoise8(_img, _h=1.0, _s=1.0, _v=0.8):
    """調整HSV通道，調整V通道以調整亮度，各通道系數"""
    _image = _img.copy()
    HSV = cv2.cvtColor(_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    H2 = np.uint8(H * _h)
    S2 = np.uint8(S * _s)
    V2 = np.uint8(V * _v)
    hsv_image = cv2.merge([H2, S2, V2])
    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return out


def addNoise8Test():
    img = cv2.imread("data/Lenna.jpg")
    out1 = addNoise8(img, _h=1, _s=1, _v=0.8)
    out2 = addNoise8(img, _h=1, _s=1, _v=1.2)
    out3 = addNoise8(img, _h=1, _s=0.8, _v=1)
    out4 = addNoise8(img, _h=1, _s=1.2, _v=1)
    out5 = addNoise8(img, _h=0.8, _s=1, _v=1)
    out6 = addNoise8(img, _h=1.2, _s=1, _v=1)
    showImage(img, out1, out2, out3, out4, out5, out6)
# endregion
# endregion