# meteor-detect

Automatic meteor detection from movie files(MP4) and streaming devices(RTSP)

kin-hasegawa さんの流星検出ソフトウェア meteor-detect の実験的 fork です。
オリジナルは https://github.com/kin-hasegawa/meteor-detect です。


## 変更点の概要

* 流星検出時にプリキャプチャー、ポストキャプチャー処理を行い、流星の出
現から消失までの完全な経過を記録します。

* 航空機・稲光の誤検出への暫定的な対策を行いました。長経路・長時間にわ
たる光跡を航空機とみなして排除することができるため、最短検出長を短く設
定しても出力があふれることがなく、より多くの流星を記録することが可能に
なりました。

* ファイル解析モード時の複数のバグを修正しました。また、動画ファイルに
格納されたメタデータとフレーム数から開始時刻を推定することで、流星の実
出現時刻を出力できるようになりました。タイムゾーンにも対応しています。

* ATOM Camに固有の機能、流星の検出と直接関係ない機能を分離(削除)しまし
た。


## コマンドラインの変更

実験的機能の実装にあたり、一部機能の簡略化と分離をおこなっており、
コマンドライン引数、オプションも若干変わっています。


```
usage: meteor-detect.py [-w] [-o OUTPUT_DIR] [-d DATETIME] [-n NAMEFORMAT]
                        [-e EXPOSURE] [-l MIN_LENGTH] [-s SIGMA] [-m MASK]
                        [--opencl] [--debug] [--help]
                        path

positional arguments:
  path                  stream URL or movie filename

optional arguments:
  -w, --show_window     show monitoring window
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        output directory
  -d DATETIME, --datetime DATETIME
                        date and time in file detection mode (ISO format)
  -n NAMEFORMAT, --nameformat NAMEFORMAT
                        format string that specifies the output filename
  -e EXPOSURE, --exposure EXPOSURE
                        exposure time (in second)
  -l MIN_LENGTH, --min_length MIN_LENGTH
                        minimum length parameter to HoughLinesP()
  -s SIGMA, --sigma SIGMA
                        sigma parameter to GaussianBlur()
  -m MASK, --mask MASK  exclusion mask
  --opencl              use OpenCL
  --debug               debug mode
  --help                show this help message and exit

```


* プログラムファイル名を atomcam.py から meteor-detect.py に変更しまし
た。

* 検出対象とするストリームURL・ファイル名をオプションではなく
positional argument として指定するようにしました。ATOM Camのストリーム
URLを仮定した暗黙指定は廃止しました。実行時にはコマンドライン上で検出
対象を必ず指定する必要があります。


```
例)
% meteor-detect rtsp://6199:4003@hostname/live
```


* この変更に伴い-u, --url, --atomcam_tools オプションを削除しました。

* -d, --datetime オプションを新設し、ファイル解析モード時おいて、動画
ファイルの撮影開始日時をコマンドラインで指定できるようにしました。メタ
データから日時を取得できないとき、手動で指定することができます。

* -n, --nameformat オプションを新設し、出力ファイル名を strftime 形式
で指定できるようにしました。

* -m, --mask オプションを拡張し、マスク範囲を数値指定できるようにしま
した。


```
例)
% meteor-detect -m "-0,1000,80,1080" rtsp://6199:4003@hostname/live
→ (0,1000)-(80,1080)で囲まれる矩形を検出除外領域とする。

% meteor-detect -m "+12,12,1908,1068" rtsp://6199:4003@hostname/live
→ (12,12)-(1908,1068)で囲まれる矩形を検出対象領域とする。

% meteor-detect -m "atomcam" rtsp://6199:4003@hostname/live
→ ATOM Camのデフォルト検出領域。(ソースコード参照)
```


* -n, --no_window オプションを論理反転し -w, --show_window オプション
としました。

* TERMシグナル受信時に検出を終了するようにしました。これにともな
い、-t, --to オプションを削除しました。

* ATOM Cam 由来のディレクトリ階層への一括アクセスを機能を削除しました。
これに伴い、-d,--date -h,--hour, -m,--minute, -i,--input オプションを
削除しました。

* --thread オプションを削除しました。

* ATOM Cam の操作を目的とした -c, --clock オプションを削除しました。
時計の同期機能は精度向上のうえ atomcam/atomsh.py に移動しました。
