import tkinter as tk
from tkinter import filedialog

import cv2


def splitVideo2Image(_input_path, _output_path, _is_gray=False):
    _cap = cv2.VideoCapture(_input_path)

    while _cap.isOpened():
        _ret, _frame = _cap.read()

        if not _ret:
            break

        if _is_gray:
            _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("splitVideo2Image", _frame)

        _key_code = cv2.waitKey(0)

        if _key_code == 27:
            break

    _cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 利用彈出視窗，選擇輸入檔案
    input_path = filedialog.askopenfilename()
    output_path = "data/split_by_action"

    print("input_path:", input_path)
    splitVideo2Image(input_path, output_path)
