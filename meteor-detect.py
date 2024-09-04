#!/usr/bin/env python

import numpy as np
import cv2 as cv

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
        self.output_dir = Path(".")
        self.nameformat = "%Y%m%d%H%M%S"
        self.debug = False
        self.show_window = False
        self.time_to = None
        self.mask = None
        self.capture = None

    def __del__(self):
        if self.capture:
            self.capture.release()
        cv.destroyAllWindows()

    # ストリームへの接続またはファイルオープン
    def connect(self):
        if self.capture:
            self.capture.release()

        self.capture = cv.VideoCapture(self.path)
        if not self.capture.isOpened():
            return False

        self.HEIGHT = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.WIDTH  = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))

        self.FPS = min(self.capture.get(cv.CAP_PROP_FPS), 60)
        # opencv-python 4.6.0.66 が大きなfps(9000)を返すことがある

        print(f"# {self.path}: ", end="")
        print(f"{self.WIDTH}x{self.HEIGHT}, {self.FPS:.3f} fps", end="")
        if self.isfile:
            total_frames = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))
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
        try:
            self.detect_meteors(exposure, min_length, sigma)
        except Exception as e:
            print(traceback.format_exc())
            self.stop()
            self.clear_image_queue()
            sys.exit(1)

        th.join()

    # 検出終了
    def stop(self):
        self._running = False

    # フレームサイズ
    def size(self):
        return (self.WIDTH, self.HEIGHT)

    # private:
    def queue_frames(self):
        tz = self.local_timezone()

        if self.isfile:
            t = self.find_file_date(self.path).astimezone(tz)
        else:
            t = datetime.now(tz)
        self.base_time = t

        print("# {} start".format(self.datetime_str(t)))
        self._running = True
        while self._running:
            r, frame = self.capture.read()
            if not r: # EOF or lost connection
                break
            self.image_queue.put((t, frame))

            if self.isfile:
                t = self.base_time + self.elapsed_time()
            else:
                t = datetime.now(tz)
                if self.time_to and self.time_to < t:
                    break

        self.image_queue.put(None)
        print("# {} stop".format(self.datetime_str(t)))

    # ファイル先頭からの経過時間
    def elapsed_time(self):
        current_pos = self.capture.get(cv.CAP_PROP_POS_FRAMES)
        current_sec = current_pos / self.FPS
        sec = int(current_sec)
        microsec = int((current_sec - sec) * 1000000)
        return timedelta(seconds=sec, microseconds=microsec)

    def detect_meteors(self, exposure, min_length, sigma):
        accmframes = [] # 流星出現直前からの累積フレーム
        postexposures = 0  # 出現後の予備露出数
        td = None  # 流星検出時刻。連続検出時は最初の検出時刻を保持。
        while True:
            # キューから exposure 秒分のフレームを取り出す
            tf = self.dequeue_frames(exposure)
            if tf is None:
                break
            (t, frames) = tf

            if self.debug:
                if t == self.base_time and self.mask is not None:
                    self.save_image(self.mask, self.timename(t) + "_m.png")
                if not self.isfile:
                    self.monitor_sky(t, frames)

            if self.show_window:
                self.show_image(frames)

            lines = self.detect_meteor_lines(frames, min_length, sigma)
            if lines is not None:
                lines = self.exclude_masked_lines(self.mask, lines)

            accmframes += frames
            if lines is not None:
                # detected frames          exposure time x 1
                if td is None:
                    td = t
                postexposures = 3        # exposure time x (2 + 1)
                # 連続検出時も postexposuresをリセット
                if self.debug:
                    self.dump_detected_lines(t, frames, lines)
            elif 1 < postexposures:
                # post-capture frames      exposure time x 2
                postexposures -= 1
            else:
                # post-capture last frames exposure time x 1
                if td is not None:
                    if not self.possible_lightning(td, accmframes) and \
                       not self.possible_airplane(td, accmframes):
                        self.detection_log(td)
                        self.save_frames(td, accmframes)
                    # 航空機・稲光判定は性質上累積フレームの情報が必要
                    td = None
                # pre-capture frames       exposure time x 0.2
                accmframes = self.last_frames(frames, 0.2)

    # 航空機対策: 10秒以上連続しているものを航空機とみなす
    # TODO: より丁寧な判定ガ必要。同時に流星も出現していたときに取り逃す。
    # 流星雨のときに困る。出現位置をもとに連続性の判定を加えるか。
    def possible_airplane(self, t, frames):
        s = len(frames) / self.FPS
        if 10.0 <= s:
            ds = self.datetime_str(t)
            print('X {} {:4.1f} sec: possible airplane'.format(ds, s))
            return True
        return False

    # 稲光対策: 輝度230以上が画面の30パーセント以上あるものを稲光とみなす
    # TODO: 世紀の大火球を取り逃すかもしれない。
    def possible_lightning(self, t, frames):
        cimage = lighten_composite(frames)
        r = brightness_rate(cimage, 230) * 100
        if 30.0 <= r:
            ds = self.datetime_str(t)
            print('X {} {:4.1f} %: possible lightning'.format(ds, r))
            return True
        return False

    # キューから exposure 秒分のフレームをとりだす
    def dequeue_frames(self, exposure):
        nframes = round(self.FPS * exposure)
        frames = []
        for n in range(nframes):
            # 'q' キー押下でプログラム終了
            if chr(cv.waitKey(1) & 0xFF) == 'q':
                self.stop()

            tf = self.image_queue.get()
            # キューから None (EOF) がでてきたら検出終了
            if tf == None:
                return None
            (tt, frame) = tf
            if n == 0:
                t = tt
            if self.opencl:
                frame = cv.UMat(frame)
            frames.append(frame)
        return (t, frames)

    def show_image(self, frames):
        cimage = lighten_composite(frames)
        cimage = self.composite_mask_to_view(cimage, self.mask)
        cv.imshow('meteor-detect: {}'.format(self.path), cimage)

    # フレーム配列の後半のfraction(0 〜 1.0)
    def last_frames(self, frames, fraction):
        n = len(frames)
        k = int(n * (1 - fraction))
        return frames[k:]

    # 線分(移動天体)を検出
    def detect_meteor_lines(self, frames, min_length, sigma):
        # (1) フレーム間の差をとり、結果を比較明合成
        diff_img = lighten_composite(diff_images(frames, None))

        # ※ オリジナル版は diff_images の第二引数に mask を与えて画像の
        # 一部をマスクしていたが、矩形マスクの縁を誤検出することがあるため
        # 判定方法を変更 -> exclude_masked_lines()

        # (2) Hough-transform で画像から線分を検出
        return detect_line_patterns(diff_img, min_length, sigma)

    # マスクされた線分を除外
    def exclude_masked_lines(self, mask, lines):
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
        cimage = lighten_composite(frames)
        self.save_image(cimage, self.timename(t) + ".jpg")
        self.save_movie(frames, self.timename(t) + ".mp4")

    # 検出結果のダンプ。デバッグ用
    def dump_detected_lines(self, t, frames, lines):
        print("D {} {} lines:".format(self.datetime_str(t), len(lines)))
        for meteor_candidate in lines:
            print('D  {}'.format(meteor_candidate))
        dimage = lighten_composite(diff_images(frames, None))
        for line in lines:
            xb, yb, xe, ye = line.squeeze()
            dimage = cv.line(dimage, (xb, yb), (xe, ye), (0, 255, 255))
        dimage = self.composite_mask_to_view(dimage, self.mask)
        self.save_image(dimage, self.timename(t) + "_d.jpg")

    # マスク領域を可視化
    def composite_mask_to_view(self, i, mask):
        if mask is None:
            return i
        return cv.addWeighted(i, 1.0, mask, 0.2, 1.0)

    # 検出ログ
    def detection_log(self, t):
        ds = self.datetime_str(t)
        if self.isfile:
            # ファイル先頭からの経過時間を付与
            et = (t - self.base_time).total_seconds()
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
        aimage = average(frames, self.opencl)
        self.save_image(aimage, self.timename(t) + "_s.jpg")

    # OpenCVで画像・動画を保存: opencv-pythonはエラー処理系がほぼザル
    def save_image(self, cimage, path):
        r = cv.imwrite(path, cimage)
        if not r:
            raise Exception('cv.imwrite', path)
    def save_movie(self, frames, path):
        video = cv.VideoWriter()
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        if not video.open(path, fourcc, self.FPS, self.size()):
            raise Exception('cv.VideoWriter.open', path)
        for frame in frames:
            video.write(frame)
        video.release()

    def timename(self, t):
        return str(Path(self.output_dir, t.strftime(self.nameformat)))

    def datetime_str(self, t):
        # ミリセカンドまで表示
        return t.strftime("%Y-%m-%d %H:%M:%S.") + t.strftime("%f")[0:3]

    def clear_image_queue(self):
        while not self.image_queue.empty():
            self.image_queue.get()

    @classmethod
    # (ビルトインな関数がありそうな気がする)
    def local_timezone(cls):
        time.tzset()
        tv = int(time.time())
        zs = tv - time.mktime(time.gmtime(tv)) # TZ offset in sec
        return timezone(timedelta(seconds=zs))

    # 動画のメタデータから日時を取得
    def find_file_date(self, u):
        try:
            date = self.mpeg_date(u)
            # Python 3.9の fromisoformat は"Z" timezoneで例外
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

'''
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
            img_list.append(cv.UMat.get(img))
    else:
        for img in list_images:
            img_list.append(img)

    return np.median(img_list, axis=0).astype(np.uint8)
'''

def average(list_images, opencl=False):
    img_list = []
    if opencl:
        for img in list_images:
            img_list.append(cv.UMat.get(img))
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
        output = cv.max(img, output)

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
            img1 = cv.bitwise_or(img1, mask)
            img2 = cv.bitwise_or(img2, mask)
        diff_list.append(cv.subtract(img1, img2))

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
    blur = cv.GaussianBlur(img, blur_size, sigma)
    canny = cv.Canny(blur, 100, 200, 3)

    # The Hough-transform algo:
    return cv.HoughLinesP(
        canny, 1, np.pi/180, 25, minLineLength=min_length, maxLineGap=5)

# 高輝度画素の比率
def brightness_rate(i, th):
    hsv = cv.cvtColor(i, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(hsv)
    n = 0
    k = 0
    for r in v.squeeze():
        for p in r.squeeze():
            n += 1
            if th <= p:
                k += 1
    return float(k)/float(n)

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
        parser.add_argument('-o', '--output_dir', default=".",
                            help='検出画像の出力先ディレクトリ名')
        parser.add_argument(
            '-n', '--nameformat', default="%Y%m%d%H%M%S",
            help="format string for output filenames in strftime format")
        parser.add_argument(
            '-m', '--mask', default=None, help="exclusion mask")
        parser.add_argument('-e', '--exposure', type=float,
                            default=1, help='露出時間(second)')
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
            if signum is not int(signal.SIGHUP):
                a.re_connect = False
            # SIGHUP かつ re_connect が Trueのとき再接続
            detector.stop()
        signal.signal(signal.SIGHUP,  signal_receptor)
        signal.signal(signal.SIGINT,  signal_receptor)
        signal.signal(signal.SIGTERM, signal_receptor)

        # 行毎に標準出力のバッファをflushする。
        sys.stdout.reconfigure(line_buffering=True)

        detector = MeteorDetect(a.path)
        detector.debug = a.debug
        detector.show_window = a.show_window
        detector.nameformat = a.nameformat
        detector.output_dir = Path(a.output_dir)
        detector.time_to = next_hhmm(a.time_to)
        while True:
            if detector.connect():
                # 接続先のフレームサイズをもとにマスク画像を生成
                detector.mask = exclusion_mask(a.mask, detector.size())
                detector.start(a.exposure, a.min_length, a.sigma)
            if not a.re_connect or detector.isfile:
                break
            # re_connect オプション指定時は5秒スリープ後に再接続
            time.sleep(5)

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
                for v in video.videostreams:
                    if str(v) == property:
                        return v.url
            except Exception as e:
                print(str(e))
            print(f"# retrying to connect to YouTube: {retry}")
            time.sleep(2)
        print(f"# {u}: retry count exceeded, exit.")
        sys.exit(1)

    # マスクフレームを生成
    def exclusion_mask(a, size):
        if a is None:
            return None

        (w, h) = size
        mask = np.zeros((h, w, 3), np.uint8)
        for t in parse_mask_text(a):
            p, bp, ep = t
            rect = np.zeros((h, w, 3), np.uint8)
            if p == '+':
                rect = cv.rectangle(rect, (0, 0), size, (255, 255, 255), -1)
                rect = cv.rectangle(rect, bp, ep, (0, 0, 0), -1)
            if p == '-':
                rect = cv.rectangle(rect, bp, ep, (255, 255, 255), -1)
            if p == '@':
                rect = cv.imread(bp)
                if rect is None:
                    raise Exception('cv.imread', bp)
            mask = lighten_composite([mask, rect])
        return mask
    # マスク指定文字列をパース
    def parse_mask_text(a):
        import re

        r = []
        for u in re.split('[ :]', a):
            if u == 'atomcam': # ATOM Cam 2/Swing
                r.append(('+', (  12,   12), (1908, 1068)))
                # 周辺10ピクセル付近にノイズが発生することがあるため、
                # 4辺の12ピクセル近傍を除外する。
                r.append(('-', (1384, 1008), (1868, 1060)))
                # 日時表示領域を除外
                continue
            if u == 'subaru': # 朝日新聞社マウナケア天文台設置カメラ
                r.append(('-', (1660,  980), (1920, 1080)))
                # 日時表示領域を除外
                continue
            if u[0] != '+' and u[0] != '-':
                r.append(('@', u, None)) # ファイル名として追加
                continue
            t = re.split(',', u[1:])
            if len(t) != 4:
                return None
            bp = (int(t[0]), int(t[1]))
            ep = (int(t[2]), int(t[3]))
            r.append((u[0], bp, ep))
        return r

    # "HHMM" 形式を現在時刻以降の最初の aware な datetimeに変換
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
