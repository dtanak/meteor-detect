#!/usr/bin/env python

import numpy as np
import cv2

from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys
import os
import time
import argparse
import threading
import queue
import traceback

class MeteorDetect:
    def __init__(self, path):
        self.path = path
        self.isfile = os.path.isfile(path)
        self.opencl = False
        self.capture = None
        self.output_dir = Path(".")
        self.basename = "%Y%m%d%H%M%S"
        self.debug = False
        self.show_window = False
        self.time_to = None

    def __del__(self):
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()

    # ストリームへの接続またはファイルオープン
    def connect(self):
        if self.capture:
            self.capture.release()

        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            return False

        self.HEIGHT = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.WIDTH  = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.FPS = min(self.capture.get(cv2.CAP_PROP_FPS), 60)
        # opencv-python 4.6.0.66 が大きなfps(9000)を返すことがある

        print(f"# {self.path}: ", end="")
        print(f"{self.WIDTH}x{self.HEIGHT}, {self.FPS:.3f} fps", end="")
        if self.isfile:
            total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            total_time = total_frames / self.FPS
            print(f", {total_frames} frames, {total_time:.1f} sec")
        else:
            print("")
        return True

    # 検出開始
    def start(self, exposure, min_length, sigma):
        self.output_dir.mkdir(exist_ok=True)

        # サブスレッドでフレームをキューに流し込む
        self.image_queue = queue.Queue(maxsize=200)
        th = threading.Thread(target=self.queue_frames)
        th.start()

        # キューからフレームを読み出し、流星を検出
        self.detect_meteors(exposure, min_length, sigma)
        self.stop()

        th.join()

    # 検出終了
    def stop(self):
        if self.image_queue:
            while not self.image_queue.empty():
                self.image_queue.get()
        self._running = False

    # フレームサイズ
    def size(self):
        return (self.WIDTH, self.HEIGHT)


    # private:
    def queue_frames(self):
        tz = self.local_timezone()

        if self.isfile:
            t = self.file_date = self.find_file_date(self.path).astimezone(tz)
        else:
            t = datetime.now(tz)
        print("# {} start".format(self.datetime_str(t)))
        self._running = True
        while self._running:
            r, frame = self.capture.read()
            if not r: # EOF or lost connection
                break
            if self.isfile:
                t = self.file_date + self.elapsed_time()
            else:
                t = datetime.now(tz)
                if self.time_to and self.time_to < t:
                    break
            self.image_queue.put((t, frame))

        self.image_queue.put(None)
        print("# {} stop".format(self.datetime_str(t)))

    # ファイル先頭からの経過時間
    def elapsed_time(self):
        current_pos = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        current_sec = current_pos / self.FPS
        sec = int(current_sec)
        microsec = int((current_sec - sec) * 1000000)
        return timedelta(seconds=sec, microseconds=microsec)

    def detect_meteors(self, exposure, min_length, sigma):
        accmframes = [] # 流星出現直前からの累積フレーム
        postexposures = 0  # 出現後の予備露出数
        detected_time = None
        while True:
            # キューから exposure 秒分のフレームを取り出す
            tf = self.dequeue_frames(exposure)
            if tf is None:
                break
            (t, frames) = tf

            if self.show_window:
                composite_img = lighten_composite(frames)
                cv2.imshow('{}'.format(self.path), composite_img)

            if not self.isfile and self.debug:
                self.monitor_sky(t, frames)

            lines = self.detect_meteor_lines(frames, min_length, sigma)
            if lines is not None:
                lines = self.mask_lines(self.mask, lines)

            accmframes += frames
            if lines is not None:
                # detected frames          exposure time x 1
                if self.debug:
                    self.dump_detected_lines(t, frames, lines)
                if detected_time is None:
                    detected_time = t
                    postexposures = 1
            elif 0 < postexposures:
                # post-capture frames      exposure time x 1
                postexposures -= 1
            else:
                # post-capture last frames exposure time x 1
                if detected_time is not None:
                    self.detection_log(detected_time)
                    self.save_frames(detected_time, accmframes)
                    detected_time = None
                # pre-capture frames       exposure time x 0.2
                accmframes = self.last_frames(frames, 0.2)

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

    # フレームのうち後半のfraction(0 〜 1.0)
    def last_frames(self, frames, fraction):
        n = len(frames)
        k = int(n * (1 - fraction))
        return frames[k:]

    # 線分(移動天体)を検出
    def detect_meteor_lines(self, frames, min_length, sigma):
        if len(frames) <= 2:
            return None
        # (1) フレーム間の差をとり、結果を比較明合成
        diff_img = lighten_composite(diff_images(frames, None))
        # ※ オリジナル版は diff_images の第二引数に mask を与えて画像の
        # 一部をマスクしていたが、矩形マスクの縁を直線として検出してしま
        # うことがあるため、線分の始点・終点がマスク領域にあるかで判定す
        # るように変更。

        # (2) Hough-transform で画像から線分を検出
        return detect_line_patterns(diff_img, min_length, sigma)

    # マスクされた線分を除外
    def mask_lines(self, mask, lines):
        r = []
        for line in lines:
            xb, yb, xe, ye = line.squeeze()
            # 始点、終点のどちらかがマスクされていなければ有効とする
            if self.valid_pos(mask, xb, yb) or self.valid_pos(mask, xe, ye):
                r.append(line)
        if len(r) == 0:
            return None
        return r
    def valid_pos(self, mask, x, y):
        if mask is None:
            return True
        pv = mask[y, x]
        # print(f"{x} {y} {pv}")
        b, g, r = pv.squeeze()
        return b == 0 and g == 0 and r == 0

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
    def save_image(self, frames, path):
        cv2.imwrite(path, lighten_composite(frames))
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

    # 検出結果のダンプ。デバッグ用
    def dump_detected_lines(self, t, frames, lines):
        print("D {} {} lines:".format(self.datetime_str(t), len(lines)))
        for meteor_candidate in lines:
            print('D  {}'.format(meteor_candidate))
        basename = t.strftime(self.basename)
        path = str(Path(self.output_dir, basename + "_d.jpg"))
        self.save_diffs(frames, path, lines)
    def save_diffs(self, frames, path, lines):
        diff = lighten_composite(diff_images(frames, None))
        for line in lines:
            xb, yb, xe, ye = line.squeeze()
            diff = cv2.line(diff, (xb, yb), (xe, ye), (0, 255, 255))
        cv2.imwrite(path, diff)

    # 検出ログ
    def detection_log(self, t):
        ds = self.datetime_str(t)
        if self.isfile:
            # ファイル先頭からの経過時間を付与
            et = (t - self.file_date).total_seconds()
            print('M {} {:8.3f}'.format(ds, et))
        else:
            # 接続先のURLを付与
            print('M {} {}'.format(ds, self.path))

    # 毎正時のスカイモニター
    def monitor_sky(self, t, frames):
        if not 'prev_monitored_time' in vars(self):
            self.prev_monitored_time = t
        if self.prev_monitored_time.hour == t.hour:
            return
        self.prev_monitored_time = t
        basename = t.strftime(self.basename)
        path = str(Path(self.output_dir, basename + "_s.jpg"))
        try:
            cv2.imwrite(path, average(frames, self.opencl))
        except Exception as e:
            print(traceback.format_exc())

    def datetime_str(self, t):
        # ミリセカンドまで表示
        return t.strftime("%Y-%m-%d %H:%M:%S.") + t.strftime("%f")[0:3]

    @classmethod
    def local_timezone(cls):
        # ローカルのタイムゾーン: ビルトインな関数がありそうな気がする。
        time.tzset()
        tv = int(time.time())
        zs = tv - time.mktime(time.gmtime(tv)) # TZ offset in sec
        return timezone(timedelta(seconds=zs))

    # ファイル日時のメタデータ
    def find_file_date(self, u):
        try:
            date = self.mpeg_date(u)
            # Python 3.9の fromisoformat は"Z" timezoneで例外を起こす。
            date = date.replace('Z', '+00:00')
            return datetime.fromisoformat(date)
        except Exception as e:
            return datetime.fromisoformat("2001-01-01T00:00:00+00:00")
    def mpeg_date(self, path):
        import subprocess
        import re

        # ffmpegコマンドを使い動画ファイルの creation_time を取得
        command = ["ffmpeg", "-i", path]
        p = subprocess.run(command, capture_output=True, text=True)
        for t in p.stderr.split("\n"):
            if "creation_time" in t:
                return re.sub('.*creation_time *: *', '', t)
        return None

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
        parser.add_argument('-w', '--show_window', action='store_true',
                            help='画面表示')
        parser.add_argument('-r', '--re_connect', action='store_true',
                            help='try to re-connect when lost connection')
        parser.add_argument('-t', '--time_to', default=None,
                            help='終了時刻(JST) "hhmm" 形式(ex. 0600)')
        parser.add_argument('-e', '--exposure', type=int,
                            default=1, help='露出時間(second)')
        parser.add_argument('-o', '--output_dir', default=".",
                            help='検出画像の出力先ディレクトリ名')
        parser.add_argument('-b', '--basename', default="%Y%m%d%H%M%S",
                            help="basename of output in strftime format")
        parser.add_argument(
            '-m', '--mask', default=None, help="mask image")
        parser.add_argument(
            '-a', '--area', default=None, help="defined detection area")
        parser.add_argument('--min_length', type=int, default=30,
                            help="minLineLength of HoghLinesP")
        parser.add_argument('--sigma', type=float, default=0.0,
                            help="sigma parameter of GaussianBlur()")
        parser.add_argument(
            '--opencl', action='store_true', help="Use OpenCL (default: False)")

        # ffmpeg関係の警告がウザいので抑制する。
        parser.add_argument(
            '-s', '--suppress-warning', action='store_true',
            help='suppress warning messages')

        parser.add_argument(
            '--debug', action='store_true', help='debug mode')
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

        # シグナルを受信したら終了
        def signal_receptor(signum, frame):
            a.re_connect = False
            detector.stop()
        signal.signal(signal.SIGHUP, signal_receptor)
        signal.signal(signal.SIGINT, signal_receptor)

        # 行毎に標準出力のバッファをflushする。
        sys.stdout.reconfigure(line_buffering=True)

        detector = MeteorDetect(a.path)
        detector.debug = a.debug
        detector.show_window = a.show_window
        detector.basename = a.basename
        detector.output_dir = Path(a.output_dir)
        detector.time_to = next_hhmm(a.time_to)
        while True:
            if detector.connect():
                # 接続先のフレームサイズをもとにマスクを生成
                detector.mask = make_mask(a.mask, a.area, detector.size())
                detector.start(a.exposure, a.min_length, a.sigma)
            if not a.re_connect or detector.isfile:
                break
            time.sleep(5)
            # re_connect オプション指定時は5秒スリープ後に再接続

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

    # マスク領域と検出領域を合成
    def make_mask(mask, area, size):
        mask = detection_mask(mask, size)
        area = detection_area(area, size)
        if mask is None:
            return area
        if area is None:
            return mask
        return lighten_composite([mask, area])

    # マスク指定を画像に変換
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

    # 検出領域指定を画像に変換
    def detection_area(a, size):
        import re

        if a is None:
            return None
        # ATOM Cam 2は周辺10ピクセル付近にノイズが発生することがあるため、
        # 4辺の12ピクセル近傍を除外する。
        if a == "atomcam":
            return detection_area("12,12-1908,1068", size)
        (w, h) = size
        zero = np.zeros((h, w, 3), np.uint8)
        mask = cv2.rectangle(zero, (0, 0), size, (255, 255, 255), -1)
        t = re.split('[,-]', a)
        bp = (int(t[0]), int(t[1]))
        ep = (int(t[2]), int(t[3]))
        return cv2.rectangle(mask, bp, ep, (0, 0, 0), -1)

    def next_hhmm(hhmm):
        if hhmm is None:
            return None
        n = datetime.now()
        t = datetime.strptime(hhmm, "%H%M")
        t = datetime(n.year, n.month, n.day, t.hour, t.minute)
        if t < n:
            t += timedelta(hours=24)
        return t.replace(tzinfo=MeteorDetect.local_timezone())

    main()
