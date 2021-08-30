import os
import subprocess
import time
from datetime import datetime

import cv2

from video.video_player import VideoPlayer


class VideoSplitter(VideoPlayer):
    def __init__(self, ffmpeg, folder: str = None):
        super().__init__()
        self.start_split = None
        self.end_split = None
        self.last_split = -1
        self.split_buffer = []
        self.writer = None
        self.s_index = 0

        self.ffmpeg = ffmpeg

        if folder is None:
            folder = "data/video"

        self.folder = folder

        self.setKeyCode()

    def setKeyCode(self):
        self.func[ord('a')] = self.startSplit
        self.func[ord('b')] = self.endSplit
        self.func[ord('c')] = self.splitBuffer
        self.func[ord('r')] = self.removeCurrentBuffer
        self.func[ord('f')] = lambda: print("lambda: ", self.cap.get(cv2.CAP_PROP_FPS))

    def setPanel(self):
        super().setPanel()

    def checkPanel(self):
        super().checkPanel()

    def startSplit(self):
        if self.last_split <= self.index:
            self.start_split = self.index
            self.last_split = self.index
            print(f"startSplit | 新的分割點({self.index})")
        else:
            print(f"startSplit | 新的分割點({self.index})，不可小於之前的分割點\n{self.split_buffer}")

    def endSplit(self):
        index = self.index + 1

        if self.last_split <= index:
            self.end_split = index
            self.last_split = index
            print(f"endSplit | 新的分割點({index})")
        else:
            print(f"endSplit | 新的分割點({index})，不可小於之前的分割點\n{self.split_buffer}")

    def splitBuffer(self):
        # ffmpeg -ss 3242.13 -i input_path -c copy -to 3561.87 output_path
        start_split = min(self.start_split, self.end_split)
        end_split = max(self.start_split, self.end_split)

        # 影片寫出
        file_name = datetime.now().strftime("%Y%m%d%H%M%S%f")
        output_path = f'{self.folder}/{file_name}.mp4'

        split_buffer = f'{self.ffmpeg} -ss {start_split / self.fps} -i "{self.path}" ' \
                       f'-c copy -to {end_split / self.fps} "{output_path}"'
        print(f"splitBuffer | 新的分割區間: {split_buffer}\n{self.split_buffer}")

        self.split_buffer.append(split_buffer)
        self.removeCurrentBuffer()

    # 當前 start_split 或 end_split 設置錯誤，重新設置當前分割點
    def removeCurrentBuffer(self):
        if self.start_split is None and self.end_split is None:
            del self.split_buffer[-1]
            print(f"removeCurrentBuffer | {self.split_buffer}")
        else:
            self.start_split = None
            self.end_split = None
            print(f"removeCurrentBuffer | start_split: {self.start_split}, end_split: {self.end_split}")

    def split(self):
        cmd = " && ".join(self.split_buffer)
        subprocess.call(cmd, shell=True)

    def split2(self):
        n_split = len(self.split_buffer)

        if n_split == 0:
            return

        self.index = 0
        self.cap = cv2.VideoCapture(self.path)

        self.s_index = 0
        start_index, end_index = self.split_buffer[self.s_index]
        self.writer, output_path = getVideoWriter(cap=self.cap, folder=self.folder)

        # 開始計時
        time_start = time.time()

        while self.cap.isOpened():
            _ret, _frame = self.cap.read()

            if not _ret:
                break
            else:
                if self.index <= end_index:

                    if start_index <= self.index:
                        # 寫入影格
                        self.writer.write(_frame)

                    self.index += 1
                else:
                    print("[Done]", output_path)

                    # 釋放所有資源
                    self.writer.release()

                    self.s_index += 1

                    if self.s_index < n_split:
                        start_index, end_index = self.split_buffer[self.s_index]
                        self.writer, output_path = getVideoWriter(cap=self.cap, folder=self.folder)
                    else:
                        break

        # 結束計時
        time_end = time.time()

        # 釋放所有資源
        print("[Done]", output_path)
        self.cap.release()
        self.writer.release()

        # 執行所花時間
        time_cost = time_end - time_start
        print('time cost', time_cost, 's')


def getVideoWriter(cap, folder="data/video", file_name: str = None, c1='m', c2='p', c3='4', c4='v'):
    # 確保資料夾存在
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 檔案名稱
    if file_name is None:
        file_name = datetime.now().strftime("%Y%m%d%H%M%S%f")

    # 影片寫出
    output_path = '{}/{}.mp4'.format(folder, file_name)

    # 取得影像的尺寸大小
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 預設使用 mp4v 編碼
    fourcc = cv2.VideoWriter_fourcc(c1, c2, c3, c4)

    # 取得影片幀率
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(output_path,
                             fourcc,
                             fps,
                             (width, height))

    return writer, output_path


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
    vs = VideoSplitter(ffmpeg=r"D:\Programing\FFMPEG\bin\ffmpeg")
    vs.openDialog()
    vs.play(use_panel=True)
    vs.split()
