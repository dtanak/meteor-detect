#!/usr/bin/env python

from pathlib import Path
import sys
import os
from datetime import datetime, timedelta, timezone
import time
import argparse
import numpy as np
import cv2
from imutils.video import FileVideoStream
import telnetlib

# 自分の環境のATOM CamのIPに修正してください。
ATOM_CAM_IP = os.environ.get("ATOM_CAM_IP", "192.168.2.110")
ATOM_CAM_RTSP = "rtsp://{}:8554/unicast".format(ATOM_CAM_IP)

# atomcam_toolsでのデフォルトのユーザアカウントなので、自分の環境に合わせて変更してください。
ATOM_CAM_USER = "root"
ATOM_CAM_PASS = "atomcam2"

class AtomTelnet():
    '''
    ATOM Camにtelnet接続し、コマンドを実行するクラス
    '''

    def __init__(self, ip_address=ATOM_CAM_IP):
        """AtomTelnetのコンストラクタ

        Args:
          ip_address: Telnet接続先のIPアドレス
        """
        self.tn = telnetlib.Telnet(ip_address)
        self.tn.read_until(b"login: ")
        self.tn.write(ATOM_CAM_USER.encode('ascii') + b"\n")
        self.tn.read_until(b"Password: ")
        self.tn.write(ATOM_CAM_PASS.encode('ascii') + b"\n")

        self.tn.read_until(b"# ")

    def exec(self, command):
        """Telnet経由でコマンドを実行する。

        Args:
          command : 実行するコマンド(ex. "ls")

        Returns:
          コマンド実行結果文字列。1行のみ。
        """
        self.tn.write(command.encode('utf-8') + b'\n')
        ret = self.tn.read_until(b"# ").decode('utf-8').split("\r\n")[1]
        return ret

    def exit(self):
        self.tn.write("exit".encode('utf-8') + b"\n")

    def __del__(self):
        self.exit()


def check_clock():
    """ATOM Camのクロックとホスト側のクロックの比較。
    """
    tn = AtomTelnet()
    atom_date = tn.exec('date')
    '''
    utc_now = datetime.now(timezone.utc)
    atom_now = datetime.strptime(atom_date, "%a %b %d %H:%M:%S %Z %Y")
    atom_now = atom_now.replace(tzinfo=timezone.utc)
    '''
    jst_now = datetime.now()
    atom_now = datetime.strptime(atom_date, "%a %b %d %H:%M:%S %Z %Y")

    dt = atom_now - jst_now
    if dt.days < 0:
        delta = -(86400.0 - (dt.seconds + dt.microseconds/1e6))
    else:
        delta = dt.seconds + dt.microseconds/1e6

    print("# ATOM Cam =", atom_now)
    print("# HOST PC  =", jst_now)
    print("# ATOM Cam - Host PC = {:.3f} sec".format(delta))


def set_clock():
    """ATOM Camのクロックとホスト側のクロックに合わせる。
    """
    tn = AtomTelnet()
    # utc_now = datetime.now(timezone.utc)
    jst_now = datetime.now()
    set_command = 'date -s "{}"'.format(jst_now.strftime("%Y-%m-%d %H:%M:%S"))
    print(set_command)
    tn.exec(set_command)

'''
class DetectMeteor():
    """
    ATOMCam 動画ファイル(MP4)からの流星の検出
    親クラスから継承したものにしたい。
    """

    def __init__(self, file_path, mask=None, minLineLength=30, opencl=False):
        # video device url or movie file path
        self.capture = FileVideoStream(file_path).start()
        self.HEIGHT = int(self.capture.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.WIDTH = int(self.capture.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.FPS = self.capture.stream.get(cv2.CAP_PROP_FPS)
        self.source = None
        self.opencl = opencl
        if self.FPS < 1.0:
            # 正しく入っていない場合があるので、その場合は15固定にする(ATOM Cam限定)。
            self.FPS = 15

        # file_pathから日付、時刻を取得する。
        # date_element = file_path.split('/')
        date_element = file_path.split(os.sep)
        self.date_dir = date_element[-3]
        self.date = datetime.strptime(self.date_dir, "%Y%m%d")

        self.hour = date_element[-2]
        self.minute = date_element[-1].split('.')[0]
        self.obs_time = "{}/{:02}/{:02} {}:{}".format(
            self.date.year, self.date.month, self.date.day, self.hour, self.minute)

        if mask:
            # マスク画像指定の場合
            self.mask = cv2.imread(mask)
        else:
            # 時刻表示部分のマスクを作成
            if opencl:
                zero = cv2.UMat((1080, 1920), cv2.CV_8UC3)
            else:
                zero = np.zeros((1080, 1920, 3), np.uint8)
            if self.source == "Subaru":
                # mask SUBRU/Mauna-Kea timestamp
                self.mask = cv2.rectangle(
                    zero, (1660, 980), (1920, 1080), (255, 255, 255), -1)
            else:
                # mask ATOM Cam timestamp
                self.mask = cv2.rectangle(
                    zero, (1390, 1010), (1920, 1080), (255, 255, 255), -1)

        self.min_length = minLineLength

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

    def meteor(self, exposure=1, output=None):
        """流星の検出
        """
        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = Path('.')

        num_frames = int(self.FPS * exposure)
        composite_img = None

        count = 0
        while self.capture.more():
            img_list = []
            for n in range(num_frames):
                try:
                    if self.capture.more():
                        frame = self.capture.read()
                        if self.opencl:
                            frame = cv2.UMat(frame)
                    else:
                        continue
                except Exception as e:
                    print(e, file=sys.stderr)
                    continue

                img_list.append(frame)

            # 画像のコンポジット
            number = len(img_list)
            count += 1

            # print(number, num_frames)
            if number > 2:
                try:
                    diff_img = brightest(diff(img_list, self.mask))
                    if detect(diff_img, self.min_length) is not None:
                        obs_time = "{}:{}".format(
                            self.obs_time, str(count*exposure).zfill(2))
                        print('{}  A possible meteor was detected.'.format(obs_time))
                        filename = self.date_dir + self.hour + \
                            self.minute + str(count*exposure).zfill(2)
                        path_name = str(Path(output_dir, filename + ".jpg"))
                        # cv2.imwrite(filename + ".jpg", diff_img)
                        composite_img = brightest(img_list)
                        cv2.imwrite(path_name, composite_img)

                        # 検出した動画を保存する。
                        movie_file = str(
                            Path(output_dir, "movie-" + filename + ".mp4"))
                        self.save_movie(img_list, movie_file)

                except Exception as e:
                    # print(traceback.format_exc(), file=sys.stderr)
                    print(e, file=sys.stderr)

def detect_meteor(args):
    """
    ATOM Cam形式の動画ファイルからの流星の検出
    """
    if args.input:
        # 入力ファイルのディレクトリの指定がある場合
        input_dir = Path(args.input)
    else:
        input_dir = Path('.')

    data_dir = Path(input_dir, args.date)
    if args.hour:
        # 時刻(hour)の指定がある場合
        data_dir = Path(data_dir, args.hour)
        if args.minute:
            # 1分間のファイル単体の処理
            file_path = Path(data_dir, "{}.mp4".format(args.minute))

    print("# {}".format(data_dir))

    if args.minute:
        # 1分間の単体のmp4ファイルの処理
        print("#", file_path)
        detecter = DetectMeteor(
            str(file_path), mask=args.mask, minLineLength=args.min_length)
        detecter.meteor(args.exposure, args.output)
    else:
        # 1時間内の一括処理
        for file_path in sorted(Path(data_dir).glob("[0-9][0-9].mp4")):
            print('#', Path(file_path))
            detecter = DetectMeteor(str(file_path), args.mask)
            detecter.meteor(args.exposure, args.output)
'''
