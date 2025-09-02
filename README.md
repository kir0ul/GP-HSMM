# GP-HSMM

ガウス過程と隠れセミマルコフモデルを用いた時系列データの分節化の実装です．詳細は以下の論文を参照してください．
This is an implementation of segmentation of time series data using Gaussian processes and hidden semi-Markov models. For details, please refer to the following paper.

Tomoaki Nakamura, Takayuki Nagai, Daichi Mochihashi, Ichiro Kobayashi, Hideki Asoh and Masahide Kaneko, “Segmenting Continuous Motions with Hidden Semi-Markov Models and Gaussian Processes”, Frontiers in Neurorobotics, vol.11, article 67, pp. 1-11, Dec. 2017 [[PDF]](https://github.com/naka-lab/GP-HSMM/raw/master/main.pdf)

さらに以下の文献で提案された高速化法を導入，計算のCython化，逆行列演算の工夫により，従来のGP-HSMMに比べ高速な計算が可能です．
Furthermore, by introducing the high-speed method proposed in the following paper, and by using Cython for the calculations and improving the inverse matrix calculations, it is possible to achieve faster calculations than the conventional GP-HSMM.

川村 美帆，佐々木 雄一，中村 裕一，"GP-HSMM の尤度計算並列化による高速な身体動作の分節化方式"，計測自動制御学会 システムインテグレーション部門講演会，1A4-08，2021
Miho Kawamura, Yuichi Sasaki, Yuichi Nakamura, "High-speed Body Motion Segmentation Method by Parallelizing Likelihood Calculation of GP-HSMM", Society of Instrument and Control Engineers, System Integration Division Conference, 1A4-08, 2021

## How to run

```
python main.py
```

Cythonで書かれたプログラムは実行時に自動的にコンパイルされます．
Programs written in Cython are automatically compiled at runtime.
WindowsのVisual Studioのコンパイラでエラーが出る場合は，
If an error occurs in the Windows Visual Studio compiler,

```
(Pythonのインストールディレクトリ)/Lib/distutils/msvc9compiler.py
(Python installation directory)/Lib/distutils/msvc9compiler.py
```

の`get_build_version()`内の
in `get_build_version()` of

```
majorVersion = int(s[:-2]) - 6
```

を使いたいVisual Studioのバージョンに書き換えてください．
Please replace with the version of Visual Studio you want to use.
VS2012の場合は，`majorVersion = 11`となります．
For VS2012, `majorVersion = 11`.

### How to use on long task

1. Get the data with:
``` sh
git clone git@github.com:PeARL-robotics/PFCS.git data/
```
2. Train the model with:
``` sh
python main.py
```
3. Apply the learned model on the PFCS data by running the `PFCS.ipynb` notebook.


# LICENSE
This program is freely available for free non-commercial use. 
If you publish results obtained using this program, please cite:

```
@article{nakamura2017segmenting,
  title={Segmenting continuous motions with hidden semi-markov models and gaussian processes},
  author={Nakamura, Tomoaki and Nagai, Takayuki and Mochihashi, Daichi and Kobayashi, Ichiro and Asoh, Hideki and Kaneko, Masahide},
  journal={Frontiers in neurorobotics},
  volume={11},
  pages={67},
  year={2017},
  publisher={Frontiers}
}
```
