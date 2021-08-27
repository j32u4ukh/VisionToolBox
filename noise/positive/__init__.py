import cv2
import numpy as np

from utils import showImage


# region # https://www.udemy.com/course/pythoncv/
# 雙邊濾波器 (Bilateral Filter):
# 是個非線性的過濾器，在計算機圖形和影像處理領域中使影像模糊化，但同時能夠保留影像內容的邊緣。
def bilateralFiltering(_img):
    dim_pixel = 7
    color = 100
    space = 100
    _filter = cv2.bilateralFilter(_img, dim_pixel, color, space)

    showImage(_img, _filter)


# 中值濾波 medianBlur:給予一個KxK大小的方形window，但是Median是找出所有所有點(最中央那個點除外)的中間值。
def medianBlur(_img):
    kernal = 3
    median = cv2.medianBlur(_img, kernal)

    showImage(_img, median)


# 高斯模糊 gaussianBlur:給予各點不同的權值，愈靠近中央點的權值愈高，最後再以平均方式計算出中央點，
# 因此，Gaussia Filter的模糊化效果比起Averaging會比較明顯，但是效果卻更為自然。
def gaussianBlur(_img):
    matrix = (7, 7)
    blur = cv2.GaussianBlur(_img, matrix, 0)

    showImage(_img, blur)


# endregion


# 均值遷移
def pyrMeanShiftFiltering(_img):
    """pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[, termcrit]]]) -> dst
    src:參數表示輸入圖像，8位，三通道圖像。
    sp:參數表示漂移物理空間半徑大小。
    sr:參數表示漂移色彩空間半徑大小。
    dst:參數表示和源圖象相同大小、相同格式的輸出圖象。
    maxLevel:參數表示金字塔的最大層數。
    termcrit:參數表示漂移叠代終止條件。"""
    dst = cv2.pyrMeanShiftFiltering(_img, 10, 30)

    showImage(_img, dst)


# region https://www.itread01.com/article/1535677121.html
"""
h：引數決定濾波器強度。較高的h值可以更好地消除噪聲，但也會刪除影象的細節 (10 is ok)
hForColorComponents：與h相同，但僅適用於彩色影象。 （通常與h相同）
templateWindowSize：應該是奇數。 （recommended 7）
searchWindowSize：應該是奇數。 （recommended 21）"""


# 單個灰度影象 去噪
def fastNlMeansDenoising(_noise):
    dst = cv2.fastNlMeansDenoising(_noise, None, 10, 7, 21)
    showImage(_noise, dst)


# 單個彩色影象 去噪
def fastNlMeansDenoisingColored(_noise):
    dst = cv2.fastNlMeansDenoisingColored(_noise, None, 10, 10, 7, 21)
    showImage(_noise, dst)


"""現在我們將相同的方法應用於視訊。
第一個引數是嘈雜幀的列表。 
第二個引數 imgToDenoiseIndex 指定我們需要去噪的幀，因為我們在輸入列表中傳遞了 frame 的索引。 
第三個是 temporalWindowSize，它指定了用於去噪的附近幀的數量。

在這種情況下，使用總共 temporalWindowSize 幀，其中中心幀是要去噪的幀。 
例如，傳遞了5個幀的列表作為輸入。設 imgToDenoiseIndex = 2 和 temporalWindowSize = 3。
然後使用 frame1、frame2 和 frame3 對 frame2 進行去噪"""


def fastNlMeansDenoisingMulti(_noise):
    # Denoise 3rd frame considering all the 5 frames
    dst = cv2.fastNlMeansDenoisingMulti(_noise, 2, 5, None, 4, 7, 35)
    showImage(dst)


# endregion


# region https://www.twblogs.net/a/5bb031222b7177781a0fd3e1
def averageFilter(_noise):
    """均值模糊 : 去隨機噪聲有很好的去噪效果
    （1, 15）是垂直方向模糊，（15， 1）是水平方向模糊"""
    vertical = cv2.blur(_noise, (1, 15))
    horizon = cv2.blur(_noise, (15, 1))
    # mix 效果較前兩者好
    mix = cv2.blur(_noise, (7, 7))
    showImage(_noise, vertical, horizon, mix)


def customBlur1(_noise):
    """用戶自定義模糊
    下面除以 25是防止數值溢出"""
    kernel = np.ones([5, 5], np.float32) / 25
    dst = cv2.filter2D(_noise, -1, kernel)

    showImage(_noise, dst)


"""高通過濾/濾波（高反差保留）
使用的函數有：cv2.Sobel() , cv2.Schar() , cv2.Laplacian()
Sobel,scharr其實是求一階或者二階導數。scharr是對Sobel的優化。
Laplacian是求二階導數。

cv2.Sobel() 是一種帶有方向過濾器
dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])
src:    需要處理的圖像；
ddepth: 圖像的深度，-1表示採用的是與原圖像相同的深度。 
        目標圖像的深度必須大於等於原圖像的深度；
dx和dy: 求導的階數，0表示這個方向上沒有求導，一般爲0、1、2。
dst     不用解釋了；
ksize： Sobel算子的大小，必須爲1、3、5、7。  ksize=-1時，會用3x3的Scharr濾波器，
        它的效果要比3x3的Sobel濾波器要好 
scale： 是縮放導數的比例常數，默認沒有伸縮係數；
delta： 是一個可選的增量，將會加到最終的dst中， 默認情況下沒有額外的值加到dst中
borderType： 是判斷圖像邊界的模式。這個參數默認值爲cv2.BORDER_DEFAULT。"""


# 嚴格來說不算去噪，只是濾波的一種
def sobel(_img):
    x = cv2.Sobel(_img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(_img, cv2.CV_16S, 0, 1)

    absx = cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    dist = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

    showImage(_img, absx, absy, dist)


# endregion
# endregion


if __name__ == "__main__":
    def bilateralFilteringTest(_mean=0, _var=0.001):
        _img = cv2.imread("data/Lenna.jpg")
        # 高斯噪音
        _noise = negative.addNoise5(_img, _mean, _var)
        bilateralFiltering(_noise)


    def medianBlurTest(_pepper=0.1):
        _img = cv2.imread("data/Lenna.jpg")
        # 中值模糊  對椒鹽噪聲有很好的去燥效果
        _noise = negative.addNoise4(_img, _pepper)
        medianBlur(_noise)


    def gaussianBlurTest(_mean=0, _var=0.001):
        _img = cv2.imread("data/Lenna.jpg")
        # 高斯噪音
        _noise = negative.addNoise5(_img, _mean, _var)
        gaussianBlur(_noise)


    def pyrMeanShiftFilteringTest(_mean=0, _var=0.001):
        _img = cv2.imread("data/Lenna.jpg")
        # 高斯噪音
        _noise = negative.addNoise5(_img, _mean, _var)
        pyrMeanShiftFiltering(_noise)


    def fastNlMeansDenoisingTest(_mean=0, _var=0.001):
        _img = cv2.imread("data/Lenna.jpg", cv2.IMREAD_GRAYSCALE)
        # 高斯噪音
        _noise = negative.addNoise5(_img, _mean, _var)
        fastNlMeansDenoising(_noise)


    def fastNlMeansDenoisingColoredTest(_mean=0, _var=0.001):
        _img = cv2.imread("data/Lenna.jpg")
        # 高斯噪音
        _noise = negative.addNoise5(_img, _mean, _var)
        fastNlMeansDenoising(_noise)


    def fastNlMeansDenoisingMultiTest():
        """效果不佳，結果畫面為全黑，理由尚不明"""
        cap = cv2.VideoCapture('data/KiMINoNaMaE.mp4')

        # create a list of first 5 frames
        frames = [cap.read()[1] for i in range(50, 55)]

        # convert all to grayscale
        gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

        # convert all to float64
        gray = [np.float64(g) for g in gray]

        # create a noise of variance 25
        noise = np.random.randn(*gray[1].shape) * 10

        # Add this noise to images
        noises = [g + noise for g in gray]

        # Convert back to uint8
        noises = [np.uint8(np.clip(n, 0, 255)) for n in noises]

        fastNlMeansDenoisingMulti(noises)


    def averageFilterTest(_mean=0, _var=0.001):
        _img = cv2.imread("data/Lenna.jpg")
        # 高斯噪音
        _noise = negative.addNoise5(_img, _mean, _var)
        averageFilter(_noise)


    def customBlur1Test(_mean=0, _var=0.001):
        _img = cv2.imread("data/Lenna.jpg")
        # 高斯噪音
        _noise = negative.addNoise5(_img, _mean, _var)
        customBlur1(_noise)


    def sobelTest():
        _img = cv2.imread("data/Lenna.jpg")
        sobel(_img)
