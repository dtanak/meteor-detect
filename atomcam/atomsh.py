#!/usr/bin/env python

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
# telnetlib は Python 3.13 で廃止予定。以降は
# https://pypi.org/project/telnetlib-313-and-up/
# を使用すること。
import telnetlib

import os
from datetime import datetime, timedelta, timezone
import time
import argparse
import re

ATOM_CAM_USER = "root"
ATOM_CAM_PASS = "atomcam2"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('host')
    parser.add_argument('command')
    args = parser.parse_args()
    if args.command == 'sync_clock' or args.command == 'comp_clock':
        exec(args.command + '(\'' + args.host + '\')')
    else:
        shell(args.host, args.command)

def shell(host, cs):
    tn = AtomTelnet(host)
    for s in tn.exec(cs)[1:-1]:
        print(s)

def sync_clock(host):
    """ATOM Camのクロックとホスト側のクロックに合わせる。
    """
    tn = AtomTelnet(host)
    p = datetime.now().strftime("%s")
    while(True):
        q = datetime.now()
        if p != q.strftime("%s"):
            break
    cs = 'date -s "{}"'.format(q.strftime("%Y-%m-%d %H:%M:%S"))
    print('# ' + tn.exec(cs)[1])

def comp_clock(host):
    """ATOM Camのクロックとホスト側のクロックの比較。
    """
    tn = AtomTelnet(host)
    jc = 'p=`date`; while true; do q=`date`; ' + \
        'if test "${p}" != "${q}"; then echo "${q}"; break; fi; done'
    for s in tn.exec(jc):
        if re.match('^(Sun|Mon|Tue|Wed|Thu|Fri|Sat)', s):
            atom_date = s
            break
    host_now = datetime.now()
    atom_now = datetime.strptime(atom_date, "%a %b %d %H:%M:%S %Z %Y")

    dt = atom_now - host_now
    if dt.days < 0:
        delta = -(86400.0 - (dt.seconds + dt.microseconds/1e6))
    else:
        delta = dt.seconds + dt.microseconds/1e6

    print("# ATOM Cam =", atom_now)
    print("# HOST PC  =", host_now)
    print("# ATOM Cam - Host PC = {:.3f} sec".format(delta))

class AtomTelnet():
    '''
    ATOM Camにtelnet接続し、コマンドを実行するクラス
    '''
    def __init__(self, ip_address="localhost"):
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
        return self.tn.read_until(b"# ").decode('utf-8').split("\r\n")

    def exit(self):
        self.tn.write("exit".encode('utf-8') + b"\n")

    def __del__(self):
        self.exit()

if __name__ == '__main__':
    main()
