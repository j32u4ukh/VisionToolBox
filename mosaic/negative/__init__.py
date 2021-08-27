import cv2

from utils import showImages


def mosaic1(img, x, y, h, w, size):
    # https://www.cnblogs.com/xdao/p/opencv_mosaic.html
    """馬賽克的實現原理是把圖像上某個像素點一定範圍鄰域內的所有點用鄰域內左上像素點的顏色代替，
    這樣可以模糊細節，但是可以保留大體的輪廓。"""
    dst = img.copy()
    if dst.ndim == 2:
        height, width = dst.shape
    else:
        height, width, _ = dst.shape

    if (y + h > height) or (x + w > width):
        return

    for i in range(0, h - size, size):
        for j in range(0, w - size, size):
            rect = [j + x, i + y, size, size]

            # 關鍵點1 tolist
            color = dst[i + y][j + x].tolist()
            left_up = (rect[0], rect[1])

            # 關鍵點2 減去一個像素
            right_down = (rect[0] + size - 1, rect[1] + size - 1)
            dst = cv2.rectangle(dst, left_up, right_down, color, -1)

    return dst


def mosaic1Test():
    img = cv2.imread("data/Lenna.jpg")
    dst = mosaic1(img, 100, 100, 150, 150, 10)
    showImages(img=img, dst=dst)


def mosaic2(img, x, y, size=10):
    X = x // size * size
    Y = y // size * size

    for i in range(size):
        for j in range(size):
            img[X + i, Y + j] = img[X, Y]

    return img


def mosaic2Test():
    # https://blog.csdn.net/weixin_38283159/article/details/78479791
    # http://www.yanglajiao.com/article/weixin_38283159/78479791
    # https://www.twblogs.net/a/5b8102152b71772165aa8cc0/

    img = cv2.imread("data/Lenna.jpg")
    enable = False

    def dynamicMosaic1(event, x, y, flags, param):
        nonlocal enable, img

        if event == cv2.EVENT_LBUTTONDOWN:
            enable = True
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if enable:
                mosaic2(img, y, x, size=10)
        elif event == cv2.EVENT_LBUTTONUP:
            enable = False

    param = None
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', dynamicMosaic1, param)

    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == 27:  # ‘esc’退出
            break

    cv2.destroyAllWindows()


def mouseEventTest():
    img = cv2.imread("data/Lenna.jpg")
    enable = False

    def mouseEvent(event, x, y, flags, param):
        # https://blog.51cto.com/devops2016/2084084
        """
        Event:
        EVENT_MOUSEMOVE 0             //滑動
        EVENT_LBUTTONDOWN 1           //左鍵點擊
        EVENT_RBUTTONDOWN 2           //右鍵點擊
        EVENT_MBUTTONDOWN 3           //中鍵點擊
        EVENT_LBUTTONUP 4             //左鍵放開
        EVENT_RBUTTONUP 5             //右鍵放開
        EVENT_MBUTTONUP 6             //中鍵放開
        EVENT_LBUTTONDBLCLK 7         //左鍵雙擊
        EVENT_RBUTTONDBLCLK 8         //右鍵雙擊
        EVENT_MBUTTONDBLCLK 9         //中鍵雙擊

        flags:
        EVENT_FLAG_LBUTTON 1       //左鍵拖曳
        EVENT_FLAG_RBUTTON 2       //右鍵拖曳
        EVENT_FLAG_MBUTTON 4       //中鍵拖曳
        EVENT_FLAG_CTRLKEY 8       //(8~15)按Ctrl不放事件
        EVENT_FLAG_SHIFTKEY 16     //(16~31)按Shift不放事件
        EVENT_FLAG_ALTKEY 32       //(32~39)按Alt不放事件"""

        nonlocal enable

        print("enable:{}".format(enable))

        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            enable = True
        elif event == cv2.EVENT_LBUTTONUP:
            enable = False
        elif event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)

    param = None
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouseEvent, param)

    while True:
        cv2.imshow('image', img)
        key_code = cv2.waitKey(1)

        if key_code == 27:
            break

    cv2.destroyAllWindows()


def mosaic3(radius=50.0):
    # https://blog.csdn.net/lm_is_dc/article/details/81412228
    img = cv2.imread("data/Lenna.jpg")
    head = img.copy()[140: 230, 130: 220]
    # head_mosaic.shape: (6, 6, 3)
    head_mosaic = head[::15, ::15]
    img_mosaic = img.copy()
    center = (3, 3)

    for x_idx in range(6):
        for y_idx in range(6):
            if pow(y_idx - center[0], 2) + pow(x_idx - center[1], 2) < pow(radius, 2):
                img_mosaic[140 + y_idx * 15: 155 + y_idx * 15, 130 + x_idx * 15: 145 + x_idx * 15] = head_mosaic[
                    y_idx, x_idx]

    showImages(img=img, img_mosaic=img_mosaic)


if __name__ == "__main__":
    mosaic3(2.5)
