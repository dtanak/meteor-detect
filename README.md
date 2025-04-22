# meteor-detect

Automatic meteor detection from movie files(MP4) and streaming devices(RTSP)

kin-hasegawa さんの流星検出ソフトウェア meteor-detect の実験的 fork です。
オリジナルは https://github.com/kin-hasegawa/meteor-detect です。


## 変更点の概要

* 流星検出時にプリキャプチャー、ポストキャプチャーを行い、出現から消失
までの完全な経過を記録します。

* 航空機・稲光の誤検出への暫定的な対策を行いました。長経路・長時間にわ
たる光跡を航空機とみなして排除することにより最短検出長を短く設定しても
出力があふれることがなくなったため、より多くの流星を記録することが可能
になりました。

* ファイル解析モード時の複数のバグを修正しました。また、動画ファイルに
格納されたメタデータとフレーム数から開始時刻を推定することで流星の実出
現時刻を出力できるようになりました。タイムゾーンにも対応しています。

* ATOM Camに固有の機能、流星の検出と直接関係ない機能を分離(削除)しまし
た。


## コマンドラインの変更

実験的機能の実装にあたり、一部機能の簡略化と分離をおこなっています。

```
usage: meteor-detect.py [-w] [-r] [--basetime BASETIME] [-o OUTPUT_DIR]
                        [-n NAMEFORMAT] [-m MASK] [-e EXPOSURE]
                        [--min_length MIN_LENGTH] [--sigma SIGMA] [--opencl]
                        [-s] [--debug] [--help]
                        path

positional arguments:
  path                  stream URL or movie filename

optional arguments:
  -w, --show_window     画面表示
  -r, --re_connect      try to re-connect when lost connection
  --basetime BASETIME   (ファイルモードの)想定開始日時
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        検出画像の出力先ディレクトリ名
  -n NAMEFORMAT, --nameformat NAMEFORMAT
                        format string for output filenames in strftime format
  -m MASK, --mask MASK  exclusion mask
  -e EXPOSURE, --exposure EXPOSURE
                        露出時間(second)
  --min_length MIN_LENGTH
                        minLineLength of HoghLinesP
  --sigma SIGMA         sigma parameter of GaussianBlur()
  --opencl              Use OpenCL (default: False)
  -s, --suppress-warning
                        suppress warning messages
  --debug               debug mode
  --help                show this help message and exit
```

* ファイル名を meteor-detect.py に変更しました。

* 検出対象とするストリームURLまたはファイル名をオプションではなく
positional argument として指定するようにしました。さらに、ATOM Cam を
仮定した暗黙指定を廃止しました。

```
例)
% meteor-detect rtsp://6199:4003@hostname/live
```

実行時には検出対象を必ず指定する必要があります。また、この変更に伴い
-u, --url, --atomcam_tools オプションを削除しました。

* --basetime オプションを新設し、ファイル解析モード時の開始時刻をコマ
ンドラインで指定できるようにしました。動画ファイルからメタデータが取得
できなかった場合に有効です。

* -n, --nameformat オプションを新設し、出力ファイル名を strftime 形式
で指定できるようにしました。

* -m, --mask オプションを拡張し、マスク範囲を数値指定できるようにしま
した。

```
例)
% meteor-detect -m "-0,1000,80,1080" rtsp://6199:4003@hostname/live
-> (0,1000)-(80,1080)で囲まれる矩形を検出除外領域とする。

% meteor-detect -m "+12,12,1908,1068" rtsp://6199:4003@hostname/live
-> (12,12)-(1908,1068)で囲まれる矩形を検出対象領域とする。

% meteor-detect -m "atomcam" rtsp://6199:4003@hostname/live
-> ATOM Camのデフォルト検出領域。
```

* -r, --re_connect オプションを新設し、コネクションが失われたときの再
接続はオプション指定時のみに行うようにしました。

* -n, --no_window オプションを論理反転し -w, --show_window オプション
としました。

* TERMシグナル受信時に検出を終了するようにしました。これにともな
い、-t, --to オプションを削除しました。

* ATOM Cam 由来の階層への一括アクセスを機能を削除しました。これに伴
い、-d,--date -h,--hour, -m,--minute, -i,--input オプションを削除しま
した。

* --thread オプションを削除しました。

* ATOM Cam の操作を目的とした -c, --clock オプションを削除しました。
時計の同期機能は精度向上のうえ atomcam/atomsh.py に移動しました。
