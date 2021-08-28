import tkinter as tk
from video import Video
from tkinter import filedialog

import cv2


class VideoPlayer(Video):
    def __init__(self):
        self.path = None
        self.cap = None
        self.index = 0
        self.frames = 0
        self.keep_playing = False

        self.is_searching = False
        self.is_playing = True

        self.speed = 1
        self.plus_more = 10
        self.minus_more = 10
        self.wait_time = 1

        self.width = 1120
        self.height = 630
        self.fps = 30.0

        self.func = {ord(' '): self.pauseOrPlay,
                     ord(','): self.minus1,
                     ord('m'): self.minusMore,
                     ord('.'): self.plus1,
                     ord('/'): self.plusMore,
                     # ESC
                     27: self.stop}

    def setKeyCode(self):
        pass

    def setPanel(self):
        # Create a window
        # cv2.namedWindow("display")

        # create trackbars for playing
        cv2.createTrackbar('speed10', "display", 0, 20, self.nothing)
        cv2.setTrackbarMin('speed10', "display", 0)
        cv2.createTrackbar('speed', "display", 1, 9, self.nothing)
        cv2.setTrackbarMin('speed', "display", 0)
        cv2.createTrackbar('jump', "display", 10, 50, self.nothing)
        cv2.setTrackbarMin('jump', "display", 2)

    def checkPanel(self):
        self.plus_more = cv2.getTrackbarPos('jump', "display")
        self.minus_more = cv2.getTrackbarPos('jump', "display")

        # getTrackbarPos(trackbarname, winname)
        speed10 = 10 * cv2.getTrackbarPos('speed10', "display")
        speed = cv2.getTrackbarPos('speed', "display")

        self.speed = speed10 + speed

    def openDialog(self):
        root = tk.Tk()
        root.withdraw()

        # 利用彈出視窗，選擇輸入檔案
        input_path = filedialog.askopenfilename()

        if input_path != "":
            self.path = input_path
            self.getFrames()

    def getFrames(self):
        self.cap = cv2.VideoCapture(self.path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames number is {self.frames}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Create a window
        cv2.namedWindow("display")

        # create trackbars for playing
        cv2.createTrackbar('progress', "display", 0, self.frames, self.getProgress)
        cv2.setTrackbarMin('progress', "display", 0)

    def play(self, use_panel=True):
        if self.path is None:
            return

        if use_panel:
            self.setPanel()

        self.index = 0
        self.cap = cv2.VideoCapture(self.path)
        self.keep_playing = True

        while self.keep_playing:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.index)
            ret, frame = self.cap.read()

            if not ret:
                break
            else:
                # 控制畫面大小
                # display = cv2.resize(_frame, (960, 540), interpolation=cv2.INTER_CUBIC)
                display = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("display", display)

                if use_panel:
                    self.checkPanel()

                # 檢查鍵盤輸入
                if self.is_playing:
                    key_code = cv2.waitKey(self.wait_time)
                else:
                    key_code = cv2.waitKey(0)

                # region Feedback to key_code
                if self.func.__contains__(key_code):
                    self.func.get(key_code)()

                elif key_code != -1:
                    print(f"key_code: {key_code} 尚未定義功能")

                # endregion

                if self.is_playing:
                    if use_panel:
                        self.checkPanel()
                    else:
                        self.speed = 1

                    if self.speed == 0:
                        self.speed = 1

                    self.index += self.speed
                    self.setProgress()

        self.cap.release()
        cv2.destroyAllWindows()

    def getProgress(self, index):
        self.index = index

    def setProgress(self):
        cv2.setTrackbarPos('progress', "display", self.index)

    def stop(self):
        self.keep_playing = False

    def plus(self, number):
        if self.index + number < self.frames:
            self.index += number
        else:
            self.index = self.frames - 1

        # print(f"index: {self.index} / {self.frames - 1}", end='\r')
        print(f"index: {self.index} / {self.frames - 1}")

    def plus1(self):
        self.plus(1)

    def plusMore(self):
        self.plus(self.plus_more)

    def minus(self, number):
        if self.index - number >= 0:
            self.index -= number
        else:
            self.index = 0

        # TODO: 或許應該呈現在 GUI 上，end='\r' 的清除速度太快，無法有效呈現
        # print(f"index: {self.index} / {self.frames - 1}", end='\r')
        print(f"index: {self.index} / {self.frames - 1}")

    def minus1(self):
        self.minus(1)

    def minusMore(self):
        self.minus(self.minus_more)

    def pauseOrPlay(self):
        # self.is_searching = not self.is_searching
        self.is_playing = not self.is_playing

    def nothing(self, value):
        # print(f"other: {value}, type: {type(value)}")
        pass


if __name__ == "__main__":
    vp = VideoPlayer()
    vp.openDialog()
    vp.play(use_panel=False)
