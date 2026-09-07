#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append("..")
from device_order import *


fig, ax = plt.subplots(figsize=(6, 4))
fig2, ax2 = plt.subplots(figsize=(6, 4))


maxbars = {}
minbars = {}

devicesToInclude = [
    "a40",
    "l40",
    "v100",
    "a100_80",
    "gh200",
    "mi210",
    "rx6900xt",
    "mi300x",
]


for filename in sorted(os.listdir("."), key=lambda f1: getOrderNumber(f1)):
    if not filename.endswith(".txt") or not any(
        [True if filename.lower().startswith(f) else False for f in devicesToInclude]
    ):
        continue
    with open(filename, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)

        mediData = {}
        maxiData = {}
        miniData = {}
        readData = {}
        triadData = {}
        initData = {}

        for row in csvreader:
            if len(row) < 18 or not row[0].isnumeric():
                continue

            print(row)
            threads = int(row[2])
            size = int(row[3])
            mini = float(row[10])
            medi = float(row[11])
            maxi = float(row[12])

            read = float(row[8])
            triad = float(row[14])
            init = float(row[17])

            if threads not in mediData:
                mediData[threads] = {}
                maxiData[threads] = {}
                miniData[threads] = {}
                readData[threads] = {}
                triadData[threads] = {}
                initData[threads] = {}

            mediData[threads][size] = medi
            maxiData[threads][size] = maxi
            miniData[threads][size] = mini
            readData[threads][size] = read
            triadData[threads][size] = triad
            initData[threads][size] = init

        # ax.scatter(
        #    [v for b in data for v in data[b].keys()],
        #    [v for b in data for v in data[b].values()],
        #    label=filename[:-4].upper(),
        #    color=getDeviceColor(filename),
        #    alpha=0.2,
        #    #   **lineStyle
        # )

        miniBWPerSize = {}
        maxBWPerSize = {}
        mediBWPerSize = {}

        for threads in mediData.keys():
            for size in mediData[threads].keys():
                if (
                    size not in mediBWPerSize
                    or mediBWPerSize[size] < mediData[threads][size]
                ):
                    maxBWPerSize[size] = maxiData[threads][size]
                    mediBWPerSize[size] = mediData[threads][size]
                    miniBWPerSize[size] = miniData[threads][size]

        ax.fill_between(
            maxBWPerSize.keys(),
            miniBWPerSize.values(),
            maxBWPerSize.values(),
            alpha=0.4,
            color=getDeviceColor(filename),
            edgecolor=None,
        )
        ax.plot(
            maxBWPerSize.keys(),
            mediBWPerSize.values(),
            color=getDeviceColor(filename),
            label=order[getOrderNumber(filename)].upper(),
            # *lineStyle,
        )
        if len(maxBWPerSize) > 0:
            ax.set_xlim([list(maxBWPerSize.keys())[0], list(maxBWPerSize.keys())[-1]])

        bws = []

        closestSize = 0
        for b in mediData.values():
            bws.append(0)
            closestSize = 0
            for v in b.items():
                if abs(v[0] - 2000) < abs(closestSize - 2000):

                    bws[-1] = v[1]
                    closestSize = v[0]

        ax2.plot(
            [k for k in mediData.keys() if k < 400000],
            bws[: len([k for k in mediData.keys() if k < 400000])],
            label=filename[:-4].upper(),
            color=getDeviceColor(filename),
            # *lineStyle,
        )

        print(closestSize)

        print(filename, getOrderNumber(filename))


########ax.set_xticks(threads[::5])
# ax.set_xticklabels(threads, rotation="vertical")
ax.set_xlabel("dataset size, MB")
ax.set_ylabel("Bandwidth, GB/s")

# ax.axhline(1400, linestyle="--", color="C1")
# ax.axhline(800, linestyle="--", color="C0")

# ax.grid()
ax.legend()
ax.set_ylim([0, ax.get_ylim()[1]])

ax.set_xscale("log", base=2)
formatter = matplotlib.ticker.FuncFormatter(
    lambda x, pos: "{0:g} kB".format(x) if x < 1024 else "{0:g}".format(x / 1024)
)
ax.get_xaxis().set_major_formatter(formatter)
ax.set_xticks(
    [
        1024,
        2048,
        4096,
        8192,
        20 * 1024,
        40 * 1024,
        96 * 1024,
        256 * 1024,
        512 * 1024,
    ]
)


fig.tight_layout()
fig.savefig("gpu-l2-stream.pdf", dpi=300)


ax2.set_xlabel("threads")
ax2.set_ylabel("Bandwidth, GB/s")


ax2.legend()
ax2.set_xlim([0, 370000])
ax2.set_ylim([0, ax2.get_ylim()[1]])

fig2.tight_layout()
fig2.savefig("gpu-l2-stream-scaling.pdf", dpi=300)


plt.show()
