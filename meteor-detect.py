#!/usr/bin/env python

import numpy as np
import cv2
from imutils.video import FileVideoStream

from pathlib import Path
import sys
import os
from datetime import datetime, timedelta, timezone
import time
import argparse
try:
    import apafy as pafy
except Exception:
    # pafyを使う場合はpacheが必要。
    import pafy

# マルチスレッド関係
import threading
import queue

import traceback

# 行毎に標準出力のバッファをflushする。
sys.stdout.reconfigure(line_buffering=True)

# YouTube ライブ配信ソース (変更になった場合は要修正)
YouTube = {
    "SDRS6JQulmI": "Kiso",
    "_8rp1p_tWlc": "Subaru",
    "ylSiGa_U1UE": "Fukushima",
    "any_youtube": "YouTube"
}

class MeteorDetect:
    def __init__(self, video_url, output=None, end_time="0600",
                 mask=None, minLineLength=30, opencl=False):
        self._running = False
        # video device url or movie file path
        self.capture = None
        self.source = None
        self.opencl = opencl
        self.isfile = os.path.isfile(video_url)

        # 入力ソースの判定
        if "youtube" in video_url:
            # YouTube(マウナケア、木曽、福島、etc)
            self.source = "YouTube"
            for source in YouTube.keys():
                if source in video_url:
                    self.source = YouTube[source]
        else:
            self.source = "ATOMCam"

        self.url = video_url

        self.connect()

        self.FPS = min(self.capture.get(cv2.CAP_PROP_FPS), 60)
        # opencv-python 4.6.0.66 が大きなfps(9000)を返すことがある

        self.HEIGHT = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.WIDTH  = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # 出力先ディレクトリ
        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = Path('.')
        self.output_dir = output_dir

        # 終了時刻を設定する。
        now = datetime.now()
        t = datetime.strptime(end_time, "%H%M")
        self.end_time = datetime(
            now.year, now.month, now.day, t.hour, t.minute)
        if now > self.end_time:
            self.end_time = self.end_time + timedelta(hours=24)

        print("# scheduled end_time = ", self.end_time)
        self.now = now

        if mask:
            # マスク画像指定の場合
            self.mask = cv2.imread(mask)
        else:
            # 時刻表示部分のマスクを作成
            if self.opencl:
                zero = cv2.UMat((1080, 1920), cv2.CV_8UC3)
            else:
                zero = np.zeros((1080, 1920, 3), np.uint8)

            if self.source == "Subaru":
                # mask SUBRU/Mauna-Kea timestamp
                self.mask = cv2.rectangle(
                    zero, (1660, 980), (1920, 1080), (255, 255, 255), -1)
            elif self.source == "YouTube":
                # no mask
                self.mask = None
            else:
                # mask ATOM Cam timestamp
                self.mask = cv2.rectangle(
                    zero, (1390, 1010), (1920, 1080), (255, 255, 255), -1)

        self.min_length = minLineLength
        self.image_queue = queue.Queue(maxsize=200)

    def __del__(self):
        now = datetime.now()
        obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )
        print("# {} stop".format(obs_time))

        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()

    def connect(self):
        if self.capture:
            self.capture.release()

        if self.source in YouTube.values():
            # YouTubeからのストリーミング入力
            video = pafy.new(self.url)
            best = video.getbest(preftype="mp4")
            url = best.url
        else:
            url = self.url

        self.capture = cv2.VideoCapture(url)

    def start(self, exposure, no_window):
        """
        RTSPストリーミング、及び動画ファイルからの流星の検出(スレッド版)
        """

        if not self.capture.isOpened():
            return

        now = datetime.now()
        obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )
        print("# {} start".format(obs_time))

        # スレッド版の流星検出
        th = threading.Thread(target=self.queue_streaming)
        th.start()

        try:
            self.dequeue_streaming(exposure, no_window)
            self.stop()
        except KeyboardInterrupt:
            self.stop()

        th.join()

    def stop(self):
        # thread を止める
        self._running = False

    def queue_streaming(self):
        """RTSP読み込みをthreadで行い、queueにデータを流し込む。
        """
        print("# threading version started.")
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._running = True
        while self._running:
            ret, frame = self.capture.read()
            if self.opencl:
                frame = cv2.UMat(frame)
            if ret:
                # self.image_queue.put_nowait(frame)
                now = datetime.now()
                self.image_queue.put((now, frame))
                if self.isfile:
                    current_pos = int(self.capture.get(
                        cv2.CAP_PROP_POS_FRAMES))
                    if current_pos >= frame_count:
                        break
            else:
                self.connect()
                time.sleep(5)

        self.image_queue.put(None)

    def dequeue_streaming(self, exposure=1, no_window=False):
        """queueからデータを読み出し流星検知、描画を行う。
        """
        num_frames = int(self.FPS * exposure)

        while True:
            img_list = []
            for n in range(num_frames):
                tf = self.image_queue.get()
                if tf is None:
                    break
                (t, frame) = tf
                key = chr(cv2.waitKey(1) & 0xFF)
                if key == 'q':
                    break

                # exposure time を超えたら終了
                if len(img_list) == 0:
                    t0 = t
                    img_list.append(frame)
                else:
                    dt = t - t0
                    if dt.seconds < exposure:
                        img_list.append(frame)
                    else:
                        break
            if len(img_list) < num_frames:
                break

            if len(img_list) > 2:
                self.composite_img = lighten_composite(img_list)
                if not no_window:
                    cv2.imshow('{}'.format(self.source), self.composite_img)
                self.detect_meteor(img_list)

            # ストリーミングの場合、終了時刻を過ぎたなら終了。
            now = datetime.now()
            if not self.isfile and now > self.end_time:
                print("# end of observation at ", now)
                self._running = False
                return

    def detect_meteor(self, img_list):
        """img_listで与えられた画像のリストから流星(移動天体)を検出する。
        """
        now = datetime.now()
        obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second)

        if len(img_list) > 2:
            # 差分間で比較明合成を取るために最低3フレームが必要。
            # 画像のコンポジット(単純スタック)
            diff_img = lighten_composite(diff_images(img_list, self.mask))
            try:
                # if True:
                if now.hour != self.now.hour:
                    # 毎時空の様子を記録する。
                    filename = "sky-{:04}{:02}{:02}{:02}{:02}{:02}".format(
                        now.year, now.month, now.day, now.hour, now.minute, now.second)
                    path_name = str(Path(self.output_dir, filename + ".jpg"))
                    mean_img = average(img_list, self.opencl)
                    # cv2.imwrite(path_name, self.composite_img)
                    cv2.imwrite(path_name, mean_img)
                    self.now = now

                detected = detect_meteor_lines(diff_img, self.min_length)
                if detected is not None:
                    '''
                    for meteor_candidate in detected:
                        print('{} {} A possible meteor was detected.'.format(obs_time, meteor_candidate))
                    '''
                    print('{} A possible meteor was detected.'.format(obs_time))
                    filename = "{:04}{:02}{:02}{:02}{:02}{:02}".format(
                        now.year, now.month, now.day, now.hour, now.minute, now.second)
                    path_name = str(Path(self.output_dir, filename + ".jpg"))
                    cv2.imwrite(path_name, self.composite_img)

                    # 検出した動画を保存する。
                    movie_file = str(
                        Path(self.output_dir, "movie-" + filename + ".mp4"))
                    self.save_movie(img_list, movie_file)
            except Exception as e:
                print(traceback.format_exc())
                # print(e, file=sys.stderr)

    def save_movie(self, img_list, pathname):
        """
        画像リストから動画を作成する。

        Args:
          imt_list: 画像のリスト
          pathname: 出力ファイル名
        """
        size = (self.WIDTH, self.HEIGHT)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        video = cv2.VideoWriter(pathname, fourcc, self.FPS, size)
        for img in img_list:
            video.write(img)

        video.release()

def composite(list_images):
    """画像リストの合成(単純スタッキング)

    Args:
      list_images: 画像データのリスト

    Returns:
      合成された画像
    """
    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)

    return output


def median(list_images, opencl=False):
    img_list = []
    if opencl:
        for img in list_images:
            img_list.append(cv2.UMat.get(img))
    else:
        for img in list_images:
            img_list.append(img)

    return np.median(img_list, axis=0).astype(np.uint8)


def average(list_images, opencl=False):
    img_list = []
    if opencl:
        for img in list_images:
            img_list.append(cv2.UMat.get(img))
    else:
        for img in list_images:
            img_list.append(img)

    return np.average(img_list, axis=0).astype(np.uint8)


def lighten_composite(img_list):
    """比較明合成処理
    Args:
      img_list: 画像データのリスト

    Returns:
      比較明合成された画像
    """
    output = img_list[0]

    for img in img_list[1:]:
        output = cv2.max(img, output)

    return output

def diff_images(img_list, mask):
    """画像リストから差分画像のリストを作成する。

    Args:
      img_list: 画像データのリスト
      mask: マスク画像(2値画像)

    Returns:
      差分画像のリスト
    """
    diff_list = []
    for img1, img2 in zip(img_list[:-2], img_list[1:]):
        if mask is not None:
            img1 = cv2.bitwise_or(img1, mask)
            img2 = cv2.bitwise_or(img2, mask)
        diff_list.append(cv2.subtract(img1, img2))

    return diff_list

def detect_meteor_lines(img, min_length):
    """画像上の線状のパターンを流星として検出する。
    Args:
      img: 検出対象となる画像
      min_length: HoughLinesPで検出する最短長(ピクセル)
    Returns:
      検出結果
    """
    blur_size = (5, 5)
    blur = cv2.GaussianBlur(img, blur_size, 0)
    canny = cv2.Canny(blur, 100, 200, 3)

    # The Hough-transform algo:
    return cv2.HoughLinesP(
        canny, 1, np.pi/180, 25, minLineLength=min_length, maxLineGap=5)

if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser(add_help=False)

        # Usage: meteor-detect [options] path

        # positional argument:
        parser.add_argument('path', help='stream URL or movie filename')

        # options:
        parser.add_argument('-n', '--no_window', action='store_true',
                            help='画面非表示')
        parser.add_argument('-e', '--exposure', type=int,
                            default=1, help='露出時間(second)')
        parser.add_argument('-o', '--output', default=None,
                            help='検出画像の出力先ディレクトリ名')
        parser.add_argument('-t', '--to', default="0600",
                            help='終了時刻(JST) "hhmm" 形式(ex. 0600)')

        parser.add_argument('--mask', default=None, help="mask image")
        parser.add_argument('--min_length', type=int, default=30,
                            help="minLineLength of HoghLinesP")

        parser.add_argument('--opencl',
                            action='store_true',
                            help="Use OpenCL (default: False)")

        # ffmpeg関係の警告がウザいので抑制する。
        parser.add_argument(
            '-s', '--suppress-warning', action='store_true',
            help='suppress warning messages')

        parser.add_argument('--help', action='help',
                            help='show this help message and exit')

        a = parser.parse_args()

        if a.suppress_warning:
            # stderrを dev/null に出力する。
            fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(fd, 2)

        detector = MeteorDetect(a.path, a.output, a.to, a.mask, a.min_length)
        detector.start(a.exposure, a.no_window)

    main()
