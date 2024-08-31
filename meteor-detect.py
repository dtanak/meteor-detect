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
import threading
import queue

import traceback

class MeteorDetect:
    def __init__(self, path, output_dir=".", end_time="0600", opencl=False):

        self.path = path
        # video device url or movie file path
        self.capture = None
        self.opencl = opencl
        self.isfile = os.path.isfile(path)
        self.output_dir = Path(output_dir)
        self.basename = "%Y%m%d%H%M%S"
        self.debug = False

        # 終了時刻を設定する。
        now = datetime.now()
        t = datetime.strptime(end_time, "%H%M")
        self.end_time = datetime(
            now.year, now.month, now.day, t.hour, t.minute)
        if now > self.end_time:
            self.end_time = self.end_time + timedelta(hours=24)

        print("# scheduled end_time = ", self.end_time)
        self.now = now

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

        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            return False

        self.FPS = min(self.capture.get(cv2.CAP_PROP_FPS), 60)
        # opencv-python 4.6.0.66 が大きなfps(9000)を返すことがある

        self.HEIGHT = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.WIDTH  = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        return True

    def start(self, exposure, min_length, sigma, no_window):
        """
        RTSPストリーミング、及び動画ファイルからの流星の検出(スレッド版)
        """

        self.output_dir.mkdir(exist_ok=True)

        now = datetime.now()
        obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )
        print("# {} start".format(obs_time))

        th = threading.Thread(target=self.queue_frames)
        th.start()
        self.detect_meteors(exposure, min_length, sigma, no_window)
        self.stop()
        th.join()

    def stop(self):
        # thread を止める
        self._running = False

    def size(self):
        return (self.WIDTH, self.HEIGHT)


    # private:
    def queue_frames(self):
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

            # ストリーミングの場合、終了時刻を過ぎたなら終了。
            now = datetime.now()
            if not self.isfile and now > self.end_time:
                print("# end of observation at ", now)
                break

        self.image_queue.put(None)

    def detect_meteors(self, exposure, min_length, sigma, no_window):
        """queueからデータを読み出し流星検知、描画を行う。
        """
        while True:
            tf = self.dequeue_frames(exposure)
            if tf is None:
                break
            (t, frames) = tf

            if not no_window:
                composite_img = lighten_composite(frames)
                cv2.imshow('{}'.format(self.path), composite_img)

            if not self.isfile and self.debug:
                monitor_sky(t, frames)

            detected = self.detect_meteor_lines(frames, min_length, sigma)
            if detected is not None:
                # self.detection_log(t)
                self.save_frames(t, frames)

    # キューから exposure 秒分のフレームをとりだす
    def dequeue_frames(self, exposure):
        nframes = round(self.FPS * exposure)
        frames = []
        for n in range(nframes):
            # 'q' キー押下でプログラム終了
            if chr(cv2.waitKey(1) & 0xFF) == 'q':
                return None

            tf = self.image_queue.get()
            # キューから None (EOF) がでてきたらプログラム終了
            if tf == None:
                return None
            (tt, frame) = tf
            if n == 0:
                t = tt
            if self.opencl:
                frame = cv2.UMat(frame)
            frames.append(frame)
        return (t, frames)

    # 毎正時のスカイモニター
    def monitor_sky(self, t, frames):
        if now.hour != self.now.hour:
            filename = "sky-{:04}{:02}{:02}{:02}{:02}{:02}".format(
                now.year, now.month, now.day, now.hour, now.minute, now.second)
            path_name = str(Path(self.output_dir, filename + ".jpg"))
            mean_img = average(img_list, self.opencl)
            # cv2.imwrite(path_name, composite_img)
            cv2.imwrite(path_name, mean_img)
            self.now = now

    # 線分(移動天体)を検出
    def detect_meteor_lines(self, frames, min_length, sigma):
        # (1) フレーム間の差をとり、結果を比較明合成
        diff_img = lighten_composite(diff_images(frames, None))

        # (2) Hough-transform で画像から線分を検出
        return detect_line_patterns(diff_img, min_length, sigma)

    # 画像・動画の保存
    def save_frames(self, t, frames):
        try:
            basename = t.strftime(self.basename)
            path_image = str(Path(self.output_dir, basename + ".jpg"))
            self.save_image(frames, path_image)
            path_movie = str(Path(self.output_dir, basename + ".mp4"))
            self.save_movie(frames, path_movie)
        except Exception as e:
            print(traceback.format_exc())
    # 画像リストを画像ファイルとして保存
    def save_image(self, frames, path):
        cv2.imwrite(path, lighten_composite(frames))
    # 画像リストを動画ファイルとして保存
    def save_movie(self, img_list, pathname):
        """
        画像リストから動画を作成する。
        Args:
          imt_list: 画像のリスト
          pathname: 出力ファイル名
        """
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(pathname, fourcc, self.FPS, self.size())
        for img in img_list:
            video.write(img)
        video.release()

    # 検出ログ
    def detection_log(self, t):
        ds = self.datetime_str(t)
        if self.isfile:
            et = (t - self.file_date).total_seconds()
            print('M {} {:8.3f}'.format(ds, et))
        else:
            print('M {} {}'.format(ds, self.path))

    def datetime_str(self, t):
        # ミリセカンドまで表示
        return t.strftime("%Y-%m-%d %H:%M:%S.") + t.strftime("%f")[0:3]

    def local_timezone(self):
        # ローカルのタイムゾーン: ビルトインな関数がありそうな気がする。
        time.tzset()
        tv = int(time.time())
        zs = tv - time.mktime(time.gmtime(tv)) # TZ offset in sec
        return timezone(timedelta(seconds=zs))


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

def detect_line_patterns(img, min_length, sigma=0):
    """画像上の線状のパターンを流星として検出する。
    Args:
      img: 検出対象となる画像
      min_length: HoughLinesPで検出する最短長(ピクセル)
    Returns:
      検出結果
    """
    blur_size = (5, 5)
    blur = cv2.GaussianBlur(img, blur_size, sigma)
    canny = cv2.Canny(blur, 100, 200, 3)

    # The Hough-transform algo:
    return cv2.HoughLinesP(
        canny, 1, np.pi/180, 25, minLineLength=min_length, maxLineGap=5)

if __name__ == '__main__':
    import argparse
    import signal

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
        parser.add_argument('-o', '--output_dir', default=".",
                            help='検出画像の出力先ディレクトリ名')
        parser.add_argument('-t', '--to', default="0600",
                            help='終了時刻(JST) "hhmm" 形式(ex. 0600)')

        parser.add_argument(
            '-m', '--mask', default=None, help="mask image")
        parser.add_argument(
            '-a', '--area', default=None, help="defined detection area")
        parser.add_argument('--min_length', type=int, default=30,
                            help="minLineLength of HoghLinesP")
        parser.add_argument('--sigma', type=float, default=0.0,
                            help="sigma parameter of GaussianBlur()")

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

        # YouTubeの場合、Full HDのビデオストリームURLを使用
        if "youtube" in a.path:
            a.path = get_youtube_stream(a.path, "video:mp4@1920x1080")

        # 行毎に標準出力のバッファをflushする。
        sys.stdout.reconfigure(line_buffering=True)

        detector = MeteorDetect(a.path, a.output_dir, a.to, a.min_length)
        try:
            if detector.connect():
                # 接続先のフレームサイズをもとにマスクを生成
                detector.mask = make_mask(a.mask, a.area, detector.size())
                detector.start(a.exposure, a.min_length, a.sigma, a.no_window)
        except KeyboardInterrupt:
            detector.stop()


    # マスク領域と検出領域を合成
    def make_mask(mask, area, size):
        mask = detection_mask(mask, size)
        area = detection_area(area, size)
        if mask is None:
            return area
        if area is None:
            return mask
        return lighten_composite([mask, area])

    # マスク領域を指定
    def detection_mask(a, size):
        import re

        if a == None:
            return None
        t = re.split('[,-]', a)
        if len(t) == 4:
            (w, h) = size
            zero = np.zeros((h, w, 3), np.uint8)
            bp = (int(t[0]), int(t[1]))
            ep = (int(t[2]), int(t[3]))
            return cv2.rectangle(zero, bp, ep, (255, 255, 255), -1)

        if a == 'atomcam': # ATOM Cam timestamp
            return detection_mask("1390,1014-1868,1056", size)
        if a == 'subaru':  # Subaru/Mauna-Kea timestamp
            return detection_mask("1660,980-1920,1080" , size)
        mask = cv2.imread(a) # 画像ファイル指定
        if mask is None:
            sys.exit(1)
        return mask

    # 検出領域を指定 (ex. --area="12,12-1884,1010")
    def detection_area(a, size):
        import re

        if a is None:
            return None
        if a == "atomcam":
            return detection_area("12,12-1908,1068", size)
        (w, h) = size
        zero = np.zeros((h, w, 3), np.uint8)
        mask = cv2.rectangle(zero, (0, 0), size, (255, 255, 255), -1)
        t = re.split('[,-]', a)
        bp = (int(t[0]), int(t[1]))
        ep = (int(t[2]), int(t[3]))
        return cv2.rectangle(mask, bp, ep, (0, 0, 0), -1)

    # YouTubeのビデオストリームURLの取得
    def get_youtube_stream(u, property):
        try:
            import apafy as pafy
        except Exception:
            # pafyを使う場合はpacheが必要。
            import pafy

        print(f"# connecting to YouTube: {u}")

        for retry in range(10):
            try:
                video = pafy.new(u, ydl_opts={'nocheckcertificate': True})
            except Exception as e:
                print(str(e))
                traceback.print_exc(file=sys.stdout)
                sys.exit(1)
            # video = pafy.new(u)
            # best = video.getbest(preftype="mp4")
            for v in video.videostreams:
                if str(v) == property:
                    return v.url
            print(f"# retrying to connect to YouTube: {retry}")
            time.sleep(2)
        print(f"# {u}: retry count exceeded, exit.")
        sys.exit(1)

    main()
