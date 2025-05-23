#!/usr/bin/env python

import numpy as np
import cv2 as cv

from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import time
import threading
import queue
import os
import sys

class MeteorDetect:
    def __init__(self, path):
        # public:
        self.isfile = os.path.isfile(path)
        self.show_window = False
        self.output_dir = Path(".")
        self.nameformat = "%Y%m%d%H%M%S"
        self.datetime = None
        self.mask = None
        self.opencl = False
        self.debug = False
        self.exposure_extension = 2.5
        # private:
        self._path = path
        self._interrupted = False
        self._capture = None

    def __del__(self):
        if self._capture:
            self._capture.release()
        cv.destroyAllWindows()

    # ストリームへの接続またはファイルオープン
    def connect(self):
        if self._capture:
            self._capture.release()

        self._capture = cv.VideoCapture(self._path)
        if self._interrupted:
            raise KeyboardInterrupt
        if not self._capture.isOpened():
            return False

        self.HEIGHT = int(self._capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.WIDTH  = int(self._capture.get(cv.CAP_PROP_FRAME_WIDTH))

        # opencv-python 4.6.0.66 が大きなfps(9000)を返すことがある
        self.FPS = min(self._capture.get(cv.CAP_PROP_FPS), 60)

        # パラメータが取得できなかったら接続失敗とみなす。
        if self.HEIGHT == 0 or self.WIDTH == 0 or self.FPS == 0:
            return False

        print(f"# {self._path}: ", end="")
        print(f"{self.WIDTH}x{self.HEIGHT}, {self.FPS:.3f} fps", end="")
        if self.isfile:
            total_frames = int(self._capture.get(cv.CAP_PROP_FRAME_COUNT))
            total_time = total_frames / self.FPS
            print(f", {total_frames} frames, {total_time:.1f} sec")
        else:
            print("")
        return True

    # 検出開始
    def start(self, exposure, min_length, sigma):
        self.output_dir.mkdir(exist_ok=True)

        # サブスレッドでフレームをキューに流し込む
        self._running_th = True
        self.image_queue = queue.Queue(maxsize=200)
        th = threading.Thread(target=self.queue_frames)
        th.start()

        try:
            # キューからフレームを読み出し、流星を検出
            self.detect_meteors(exposure, min_length, sigma)
        finally:
            # 例外終了時もサブスレッドを確実に終了させる。
            self.stop()
            self.clear_queue() # image_queue.put()のブロックを回避
            th.join()

    # 検出終了
    def stop(self):
        self._running_th = False

    # 検出中断
    def user_interrupt(self):
        self._interrupted = True

    def size(self):
        return (self.WIDTH, self.HEIGHT)

    # private:
    def queue_frames(self):
        tz = self.local_tz()

        if self.isfile:
            if self.datetime is None:
                self.datetime = datetime.fromisoformat("0001-01-01T00:00:00")
            self.datetime = self.apply_tz(self.datetime, tz)
        else:
            self.datetime = datetime.now(tz)

        t = self.datetime
        while self._running_th:
            r, frame = self._capture.read()
            if not r: # EOF or lost connection
                break
            self.image_queue.put((t, False, frame))

            if self.isfile:
                t = self.datetime + self.elapsed_time()
            else:
                t = datetime.now(tz)

        self.image_queue.put((t, True, None))

    # ファイル先頭からの経過時間
    def elapsed_time(self):
        current_pos = self._capture.get(cv.CAP_PROP_POS_FRAMES)
        current_sec = current_pos / self.FPS
        sec = int(current_sec)
        microsec = int((current_sec - sec) * 1000000)
        return timedelta(seconds=sec, microseconds=microsec)

    def detect_meteors(self, exposure, min_length, sigma):
        accmframes = [] # 流星出現直前からの累積フレーム
        td = None       # 流星検出時刻。
        exposure_ = exposure
        while True:
            # キューから exposure_ 秒分のフレームを取り出す
            (t, eod, frames) = self.dequeue_frames(exposure_)
            if eod is True:
                break
            accmframes += frames

            if t == self.datetime:
                print("# {} start".format(self.datetime_str(t)))

            if self.show_window:
                self.show_image(frames)

            lines = self.detect_meteor_lines(frames, min_length, sigma)
            if lines is not None:
                if td is None:
                    td = t
                    exposure_ = exposure * self.exposure_extension
                    # ひとたび線分を検出したら、exposure を延長することで
                    # 条件を緩めて検出を継続する。
                if self.debug:
                    self.dump_detected_lines(t, frames, lines)
            else:
                if td is not None:
                    # 線分の検出が途切れたら判定を行いフレームを保存。
                    if self.possible_meteor(td, accmframes):
                        self.detection_log(td)
                        self.save_frames(td, accmframes)
                    td = None
                accmframes = self.precapture(frames, 0.5)
                exposure_ = exposure
        print("# {} stop".format(self.datetime_str(t)))

    def possible_meteor(self, t, frames):
        return not self.possible_aircraft(t, frames) and \
            not self.possible_lightning(t, frames)

    # 航空機判定: 7秒以上連続したものを航空機とみなす
    # TODO: 航空機と同時に出現した流星を取り逃す。複数の流星の連続出現を
    # 誤判定する可能性がある。出現位置をもとした連続性の判定を加えるか。
    def possible_aircraft(self, t, frames):
        s = len(frames) / self.FPS
        if 7.0 <= s:
            ds = self.datetime_str(t)
            print(f"X {ds} {self._path} {s:4.1f} sec: possible aircraft")
            return True
        return False

    # 稲光判定: 輝度230以上が画面の30パーセント以上あるものを稲光とみなす
    # TODO: 稲光に匹敵するような大火球を取り逃す可能性がある。
    def possible_lightning(self, t, frames):
        cimage = lighten_composite(frames)
        r = brightness_rate(cimage, 230) * 100
        if 30.0 <= r:
            ds = self.datetime_str(t)
            print(f"X {ds} {self._path} {r:4.1f} %: possible lightning")
            return True
        return False

    # キューから exposure 秒分のフレームをとりだす
    def dequeue_frames(self, exposure):
        nframes = round(self.FPS * exposure)
        frames = []
        for n in range(nframes):
            # 'q' キー押下やシグナルの受信も KeyboardInterrupt として扱う
            if chr(cv.waitKey(1) & 0xFF) == 'q' or self._interrupted:
                raise KeyboardInterrupt

            (tt, eod, frame) = self.image_queue.get()
            if eod is True: # 検出終了
                return (tt, True, None)
            if n == 0:
                t = tt
            if self.opencl:
                frame = cv.UMat(frame)
            frames.append(frame)
        return (t, False, frames)

    def clear_queue(self):
        while not self.image_queue.empty():
            self.image_queue.get()

    def show_image(self, frames):
        # pimage = average(frames)
        pimage = lighten_composite(frames)
        pimage = self.composite_mask_to_view(pimage, self.mask)
        cv.imshow('meteor-detect: {}'.format(self._path), pimage)

    # プレキャプチャー: framesの最後からsec秒分のフレームを返す。
    def precapture(self, frames, sec):
        n = len(frames)
        k = (int)(n - sec * self.FPS)
        if k < 0:
            k = 0
        return frames[k:]

    # 線分(移動天体)を検出
    def detect_meteor_lines(self, frames, min_length, sigma):
        # (1) フレーム間の差をとり、結果を比較明合成
        diff_img = lighten_composite(diff_images(frames, None))

        # ※ オリジナル版は diff_images の第二引数に mask を与えて画像の
        # 一部をマスクしていたが、矩形マスクの縁を誤検出することがあるため
        # 判定方法を変更 -> exclude_masked_lines()

        # (2) Hough-transform で画像から線分を検出
        lines = detect_line_patterns(diff_img, min_length, sigma)

        # (3) マスク領域のチェック。始点・終点のどちらかが領域外にあれば有効
        if lines is not None:
            lines = self.exclude_masked_lines(self.mask, lines)
        return lines

    # マスクされた線分を除外
    def exclude_masked_lines(self, mask, lines):
        r = []
        for line in lines:
            xb, yb, xe, ye = line.squeeze()
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

    # 検出動画・画像の保存
    def save_frames(self, t, frames):
        cimage = lighten_composite(frames)
        self.save_image(cimage, self.timename(t) + ".jpg")
        self.save_movie(frames, self.timename(t) + ".mp4")

    # 検出ログ
    def detection_log(self, t):
        ds = self.datetime_str(t)
        if self.isfile:
            # ファイル先頭からの経過時間を付与
            et = (t - self.datetime).total_seconds()
            print('M {} {:8.3f}'.format(ds, et))
        else:
            # 接続先のURLを付与
            print('M {} {}'.format(ds, self._path))

    # 検出結果のダンプ。デバッグ用
    def dump_detected_lines(self, t, frames, lines):
        ds = self.datetime_str(t)
        n = len(lines)
        print(f"D {ds} {self._path} {n} lines:")
        for line in lines:
            xb, yb, xe, ye = line.squeeze()
            length = math.sqrt((xe - xb) **2 + (ye - yb) **2)
            print(f"D  {line} {length:.1f}")
        dimage = lighten_composite(diff_images(frames, None))
        for line in lines:
            xb, yb, xe, ye = line.squeeze()
            dimage = cv.line(dimage, (xb, yb), (xe, ye), (0, 255, 255))
        dimage = self.composite_mask_to_view(dimage, self.mask)
        self.save_image(dimage, self.timename(t) + "_d.jpg")

    # マスク領域を合成して可視化
    def composite_mask_to_view(self, i, mask):
        if mask is None:
            return i
        return cv.addWeighted(i, 1.0, mask, 0.2, 1.0)

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

    # 出力ファイル名の共通部分
    def timename(self, t):
        return str(Path(self.output_dir, t.strftime(self.nameformat)))

    # 出力ログ用の日時部分。ミリセカンドまで表示
    def datetime_str(self, t):
        return t.strftime("%Y-%m-%d %H:%M:%S.") + t.strftime("%f")[0:3]

    @classmethod
    def local_tz(cls):
        time.tzset()
        tv = int(time.time())
        zs = tv - time.mktime(time.gmtime(tv)) # TZ offset in sec
        return timezone(timedelta(seconds=zs))
    # TODO: きちんとしたローカルタイムゾーン取得方法があるはず。
    # UTCとの秒数差だけというのは情報として不十分、たとえば夏時間の運用は
    # わからない。

    @classmethod
    def apply_tz(cls, t, tz):
        # datetime に timezone を適用
        # タイムゾーン情報なし(naive): tz を付与
        # タイムゾーン情報あり(aware): tz で換算
        if t.tzinfo is None:
            return t.replace(tzinfo=tz)
        else:
            return t.astimezone(tz)

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
        a = parse_arguments()

        # 行毎に標準出力のバッファをflushする。
        sys.stdout.reconfigure(line_buffering=True)

        # YouTubeの場合、FHDのビデオストリームURLを使用
        if "youtube" in a.path:
            a.path = get_youtube_stream(a.path, "video:mp4@1920x1080")

        detector = MeteorDetect(a.path)

        detector.debug = a.debug
        detector.show_window = a.show_window
        detector.output_dir = Path(a.output_dir)
        detector.nameformat = a.nameformat
        if os.path.isfile(a.path):
            detector.datetime = find_datetime_in_metadata(a.path)
        if a.datetime:
            detector.datetime = datetime.fromisoformat(a.datetime)

        def signal_receptor(signum, frame):
            detector.user_interrupt()
        signal.signal(signal.SIGTERM, signal_receptor)

        # ソースへの接続
        if not detector.connect():
            sys.exit(1)

        # 接続先のフレームサイズをもとにマスク画像を生成
        detector.mask = make_exclusion_mask(a.mask, detector.size())

        # 検出開始(接続断後の再接続はしない)
        detector.start(a.exposure, a.min_length, a.sigma)

    def parse_arguments():
        parser = argparse.ArgumentParser(add_help=False)

        # Usage: meteor-detect [options] path

        # positional argument:
        parser.add_argument('path', help='stream URL or movie filename')

        # options:
        parser.add_argument(
            '-w', '--show_window', action='store_true',
            help='show monitoring window')
        parser.add_argument(
            '-o', '--output_dir', default=".",
            help='output directory')
        parser.add_argument(
            '-d', '--datetime', default=None,
            help='date and time in file detection mode (ISO format)')
        parser.add_argument(
            '-n', '--nameformat', default="%Y%m%d%H%M%S",
            help="format string that specifies the output filename")
        parser.add_argument(
            '-e', '--exposure', type=float, default=1,
            help='exposure time (in second)')
        parser.add_argument(
            '-l', '--min_length', type=int, default=15,
            help="minimum length parameter to HoughLinesP()")
        parser.add_argument(
            '-s', '--sigma', type=float, default=0.0,
            help="sigma parameter to GaussianBlur()")
        parser.add_argument(
            '-m', '--mask', default=None,
            help="exclusion mask")

        parser.add_argument(
            '--opencl', action='store_true',
            help="use OpenCL")
        parser.add_argument(
            '--debug', action='store_true',
            help='debug mode')
        parser.add_argument(
            '--help', action='help',
            help='show this help message and exit')

        return parser.parse_args()

    def find_datetime_in_metadata(path):
        import subprocess
        import re

        try:
            # ffmpegコマンドで動画ファイルの creation_time を取得
            command = ["ffmpeg", "-i", path]
            p = subprocess.run(command, capture_output=True, text=True)
            for s in p.stderr.split("\n"):
                if "creation_time" in s:
                    ts = re.sub('.*creation_time *: *', '', s)
                    # "Z" timezone は Python 3.9 がパースできない。
                    ts = ts.replace('Z', '+00:00')
                    return datetime.fromisoformat(ts)
        except:
            pass
        return None

    # YouTubeのビデオストリームURLの取得
    def get_youtube_stream(u, property):
        try:
            import apafy as pafy
        except Exception:
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

    def make_exclusion_mask(a, size):
        if a is None:
            return None

        (w, h) = size
        mask = np.zeros((h, w, 3), np.uint8)
        for t in parse_mask_text(a):
            p, bp, ep = t
            rect = np.zeros((h, w, 3), np.uint8)
            if p == '+':
                # detection area
                rect = cv.rectangle(rect, (0, 0), size, (255, 255, 255), -1)
                rect = cv.rectangle(rect, bp, ep, (0, 0, 0), -1)
            if p == '-':
                # exclusion area
                rect = cv.rectangle(rect, bp, ep, (255, 255, 255), -1)
            if p == '@':
                # mask-image file
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
                # r.append(('-', (1384, 1008), (1868, 1060)))
                r.append(('-', (1380, 1000), (1920, 1080)))
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

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
