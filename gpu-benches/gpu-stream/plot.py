#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append("..")
from device_order import *

# fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor="w", figsize=(8, 5))
fig, ax = plt.subplots(figsize=(6, 4))

# fig2, ax2 = plt.subplots(figsize=(8, 4))
# fig3, ax3 = plt.subplots(figsize=(8, 4))


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
    # "mi300a",
]


for filename in sorted(os.listdir("."), key=lambda f1: getOrderNumber(f1)):
    if not filename.endswith(".txt") or not any(
        [True if filename.lower().startswith(f) else False for f in devicesToInclude]
    ):
        continue
    with open(filename, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        threads = []
        locs = []
        init = []
        read = []
        scale = []
        triad = []
        stencil3pt = []
        stencil5pt = []

        for row in csvreader:
            if row[0].startswith("block") or len(row) < 12:
                continue

            # print(row)
            threads.append(int(row[1]))
            init.append(float(row[6]))
            read.append(float(row[7]))
            scale.append(float(row[8]))
            triad.append(float(row[9]))
            locs.append(float(row[2]))
            stencil3pt.append(float(row[10]))
            stencil5pt.append(float(row[11]))

        if len(threads) < 1:
            continue

        # locs = threads#[15 + l / 6 if l > 15 else l for l in locs]
        # print(locs)
        # print(threads)
        # ax.plot(locs, init,  "-v", label=filename, color="C" + str(color))
        ax.plot(
            np.array(threads),
            scale,
            label=order[getOrderNumber(filename)].upper(),
            color=getDeviceColor(filename),
            **lineStyle
        )
        # ax2.plot(
        #    np.array(threads),
        #    triad,
        #    label=filename[:-4].upper(),
        #    color=getDeviceColor(filename),
        #    **lineStyle
        # )
        print(filename, getOrderNumber(filename))

        # ax.plot(threads, triad, "-<", label=filename, color="C" + str(color))
        # ax.plot(threads, read, "-^", label=filename, color="C" + str(color))

        maxbars[filename] = [
            read[-1],
            scale[-1],
            triad[-1],
            init[-1],
            # stencil3pt[-1],
            # stencil5pt[-1],
        ]

        mClosest = 0
        for m in range(len(threads)):
            if abs(threads[m] - 10000) < abs(threads[mClosest] - 10000):
                mClosest = m

        print(threads[mClosest])
        minbars[filename] = [
            read[mClosest],
            scale[mClosest],
            triad[mClosest],
            init[mClosest],
            # stencil3pt[0],
            # stencil5pt[0],
        ]


########ax.set_xticks(threads[::5])
# ax.set_xticklabels(threads, rotation="vertical")
ax.set_xlabel("threads")
ax.set_ylabel("DRAM bandwidth, GB/s")

# ax.axhline(1400, linestyle="--", color="C1")
# ax.axhline(800, linestyle="--", color="C0")

# ax.grid()
#
#
# ax.set_xscale("log")
ax.legend()

ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlim([0, 400000])

formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: "{:.0f}K".format(x // 1000))
ax.get_xaxis().set_major_formatter(formatter)

fig.tight_layout(pad=0)
fig.savefig("cuda-stream.svg", dpi=300)
fig.savefig("cuda-stream.pdf", dpi=300)


plt.show()

print(maxbars)


def plotXbars(xbars, filename):
    fig2, ax2 = plt.subplots(figsize=(6, 3))

    valueCount = len(list(xbars.values())[0])
    c = 0
    for m in range(valueCount):
        ax2.bar(
            np.arange(len(xbars))
            + 0.8
            / valueCount
            * (m + 0.5 - valueCount / 2),  # + (0.9 * valueCount)  - 0.5,
            [i[m] for i in xbars.values()],
            width=0.8 / valueCount,
            color=device_color_palette[c],
            label=["read", "scale", "triad", "init", "1D3PT", "1D5PT"][m],
        )
        # for n in range(len(maxbars)):
        #    ax2.text(
        #        n + 0.9 * (m - 0.5) / valueCount - 0.35,
        #        150,
        #        ["init", "read", "scale", "triad", "1D3PT", "1D5PT"][m],
        #        rotation=90,
        #        color="w",
        #        horizontalalignment="left",
        #    )
        c += 1

    # ax2.text(-0.4, 51, "init", rotation=90, color="w")
    # ax2.text(-0.28, 51, "read", rotation=90, color="w")
    # ax2.text(-0.16, 51, "scale", rotation=90, color="w")
    # ax2.text(-0.04, 51, "triad", rotation=90, color="w")
    # ax2.text(0.08, 51, "1D3PT", rotation=90, color="w")
    # ax2.text(0.22, 51, "1D5pt", rotation=90, color="w")

    print(list(maxbars.keys()))
    ax2.set_xticks(range(len(list(maxbars.keys()))))
    ax2.set_xticklabels(
        [order[getOrderNumber(f)].upper() for f in list(maxbars.keys())]
    )
    ax2.set_ylabel("DRAM Bandwidth, GB/s")
    fig2.autofmt_xdate()
    ax2.legend()
    fig2.tight_layout(pad=0)
    fig2.savefig(filename, dpi=300)
    plt.show()


plotXbars(maxbars, "maxbars.pdf")
plotXbars(minbars, "minbars.pdf")
