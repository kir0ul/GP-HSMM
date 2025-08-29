# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from GPSegmentation import GPSegmentation
import time
from pathlib import Path
import numpy as np
import shutil

# DIM = 2
# NCLASS = 5
# DIM = 2
NCLASS = 5

# MAX_LEN=20
# MIN_LEN=5
# AVE_LEN=10
# SKIP_LEN=1


def conv_time2ptsnb(time_min, time_max, freq):
    dt = 1 / freq
    max_len = int(np.round(time_max / dt))
    min_len = int(np.round(time_min / dt))
    avg_len = int(np.round(np.mean([max_len, min_len])))
    skip_len = int(np.round(min_len / 2))
    print(f"MAX_LEN: {max_len}")
    print(f"MIN_LEN: {min_len}")
    print(f"AVG_LEN: {avg_len}")
    print(f"SKIP_LEN: {skip_len}")
    return max_len, min_len, avg_len, skip_len


MAX_LEN, MIN_LEN, AVE_LEN, SKIP_LEN = conv_time2ptsnb(
    time_min=60, time_max=120, freq=45
)

# data_path = Path(".")
# data_path = Path(".") / "data" / "fetch_table_demos"
# data_path = Path(".") / "data" / "LASADataset"
data_path = Path(".") / "data" / "PFCS" / "table task" / "export"

# files =  [ "testdata2d_%03d.txt" % j for j in range(4) ]
# files =  [ data_path / f"BendedLine_positions_{idx}.txt" for idx in range(7) ]
# files =  [ data_path / f"BendedLine_positions_concat_{idx}.txt" for idx in range(1, 3) ]
files = list(data_path.glob("fetch*.txt"))

# files =  [ data_path / "BendedLine_positions_concat.txt" ]
data_dimensions = None
for fname in files:
    print(fname.absolute())
    data = np.loadtxt(fname)
    if data_dimensions is None:
        data_dimensions = data.shape
    if data.shape != data_dimensions:
        raise ValueError("All the data files don't have the same dimensions")
DIM = data.shape[1]
print(f"DIM = {DIM}")

learn_path = data_path / "learn"
recog_path = data_path / "recog"
if learn_path.exists():
    shutil.rmtree(learn_path)
if recog_path.exists():
    shutil.rmtree(recog_path)


def learn(savedir):
    # gpsegm = GPSegmentation(dim=2, nclass=5)
    # gpsegm = GPSegmentation(dim=DIM, nclass=NCLASS)
    gpsegm = GPSegmentation(
        dim=DIM,
        nclass=NCLASS,
        MAX_LEN=MAX_LEN,
        MIN_LEN=MIN_LEN,
        AVE_LEN=AVE_LEN,
        SKIP_LEN=SKIP_LEN,
    )

    # files =  [ "testdata2d_%03d.txt" % j for j in range(4) ]
    gpsegm.load_data(files)

    start = time.time()
    it_num = 5
    for it in range(it_num):
        print(f"----- Iteration: {it}/{it_num} -----")
        gpsegm.learn()
        gpsegm.save_model(savedir)
        print("lik =", gpsegm.calc_lik())
    print(time.time() - start)
    return gpsegm.calc_lik()


def recog(modeldir, savedir):
    # gpsegm = GPSegmentation(dim=2, nclass=5)
    # gpsegm = GPSegmentation(dim=DIM, nclass=NCLASS)
    gpsegm = GPSegmentation(
        dim=DIM,
        nclass=NCLASS,
        MAX_LEN=MAX_LEN,
        MIN_LEN=MIN_LEN,
        AVE_LEN=AVE_LEN,
        SKIP_LEN=SKIP_LEN,
    )

    # files = [ "testdata2d_%03d.txt" % j for j in range(4) ]
    gpsegm.load_data(files)
    gpsegm.load_model(modeldir)

    start = time.time()
    gpsegm.recog()
    print("lik =", gpsegm.calc_lik())
    print(time.time() - start)
    gpsegm.save_model(savedir)


def main():
    # learn( "learn/" )
    # recog( "learn/" , "recog/" )
    learn(learn_path)
    recog(learn_path, recog_path)
    return


if __name__ == "__main__":
    main()
