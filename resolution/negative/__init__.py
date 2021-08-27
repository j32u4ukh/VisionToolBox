import cv2

from utils import showImage


def resizeTest(_width_scale, _height_scale):
    # https://jennaweng0621.pixnet.net/blog/post/403862273-%5Bpython-%2B-
    # opencv%5D-%E8%AA%BF%E6%95%B4%E5%BD%B1%E5%83%8F%E5%A4%A7%E5%B0%8F%28resize%29
    _img = cv2.imread("../../OpenEyes/data/splice4.png")
    _rows, _cols, _ = _img.shape

    # rows:1440, cols:1080
    print("rows:{}, cols:{}".format(_rows, _cols))

    _resize_rows = int(_rows * _height_scale)
    _resize_cols = int(_cols * _width_scale)
    INTER_NEAREST = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_NEAREST)
    INTER_LINEAR = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_LINEAR)
    INTER_AREA = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_AREA)
    INTER_CUBIC = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_CUBIC)
    INTER_LANCZOS4 = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_LANCZOS4)

    print("INTER_LANCZOS4.shape:{}".format(INTER_LANCZOS4.shape))

    showImage(_img, INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4)


def pyrDown(_width_scale=0.5, _height_scale=0.5):
    # 高斯金字塔
    # https://blog.csdn.net/on2way/article/details/46867939
    _img = cv2.imread("../../OpenEyes/data/sk_image1.png", cv2.IMREAD_GRAYSCALE)
    _rows, _cols = _img.shape
    print("rows:{}, cols:{}".format(_rows, _cols))
    _new_rows = int(_rows * _height_scale)
    _new_cols = int(_cols * _width_scale)
    _down_img = cv2.pyrDown(_img, dstsize=(_new_cols, _new_rows))
    print("_down_img.sahpe:{}".format(_down_img.shape))

    _up_img = cv2.pyrUp(_down_img, dstsize=(_cols, _rows))
    print("_up_img.sahpe:{}".format(_up_img.shape))

    showImage(_img, _down_img, _up_img)


def pyrDown2():
    # https://blog.csdn.net/on2way/article/details/46867939
    # 拉普拉斯金字塔的圖像看起來就像是邊界圖，經常被用在圖像壓縮中。
    _img = cv2.imread("../../OpenEyes/data/pyrDown1.png", cv2.IMREAD_GRAYSCALE)

    _down_img = cv2.pyrDown(_img)  # 高斯金字塔
    print("_down_img.sahpe:{}".format(_down_img.shape))

    _down_down_img = cv2.pyrDown(_down_img)
    print("_down_down_img.sahpe:{}".format(_down_down_img.shape))
    _up_down_img = cv2.pyrUp(_down_down_img)
    print("_up_down_img.sahpe:{}".format(_up_down_img.shape))
    _laplace = _down_img - _up_down_img
    print("_laplace.sahpe:{}".format(_laplace.shape))

    showImage(_img, _down_img, _laplace)

