import cv2

from neutral.utils import showImage


def addOcclude1():
    # 我自己根據之前學的畫圖方式，加上一些干擾
    _img = cv2.imread("data/Lenna.jpg")
    _rows, _cols, _ = _img.shape
    _half_rows = int(_rows / 2)
    _half_cols = int(_cols / 2)

    # rectangle
    cv2.rectangle(_img,
                  (0, 0),
                  (_cols, _half_rows),
                  (123, 200, 98),
                  5,
                  lineType=8,
                  shift=0)
    # line
    cv2.line(_img,
             (_half_cols + 50, _half_rows + 50),
             (_cols, _half_rows + 10),
             (0, 0, 255),
             5)

    # circle
    _color = (255, 0, 255)
    cv2.circle(_img,
               (_half_cols, _half_rows),
               50,
               _color,
               5)

    # text
    _font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(_img,
                "OpenCV",
                (100, 100),
                _font,
                1,
                (255, 255, 255),
                1,
                cv2.LINE_8)

    showImage(_img)


if __name__ == "__main__":
    addOcclude1()
