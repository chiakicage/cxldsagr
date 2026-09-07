#!/usr/bin/env python3

import os
import csv

import sys

sys.path.append("..")
from device_order import *


devicesToInclude = []


if len(sys.argv) > 1 and sys.argv[1] == "AMD":
    devicesToInclude = ["MI100", "MI210", "MI300X", "RX6900XT"]

if len(sys.argv) > 1 and sys.argv[1] == "NV":
    devicesToInclude = ["A40", "L40", "V100", "A100", "GH200"]


fig, ax = plt.subplots(figsize=(6, 4))
for filename in sorted(os.listdir("."), key=lambda f1: getOrderNumber(f1)):
    if not filename.endswith(".txt") or getOrderNumber(filename) > len(order):
        continue
    if len(devicesToInclude) > 0 and not any(
        [filename.upper().startswith(d) for d in devicesToInclude]
    ):
        continue

    with open(filename, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        sizes = []
        min = []
        max = []
        avg = []
        med = []
        for row in csvreader:
            if len(row) < 8 or row[0] == "clock:":
                continue
            sizes.append(float(row[2]))
            avg.append(float(row[4]))
            med.append(float(row[5]))
            min.append(float(row[6]))
            max.append(float(row[7]))

        print(filename, getOrderNumber(filename))

        ax.plot(
            sizes,
            med,
            # "-x",
            label=order[getOrderNumber(filename)].upper(),
            color=getDeviceColor(filename),
            **lineStyle
        )

        plt.fill_between(
            sizes, min, max, alpha=0.4, color=getDeviceColor(filename), edgecolor=None
        )


ax.set_xlabel("chain data volume, kB")
ax.set_ylabel("latency, cycles")
ax.set_xscale("log", base=2)


# ax.axvline(16)
# ax.axvline(4*1024)

formatter = matplotlib.ticker.FuncFormatter(
    lambda x, pos: "{0:g} kB".format(x) if x < 1024 else "{0:g} MB".format(x // 1024)
)
ax.get_xaxis().set_major_formatter(formatter)
# ax.get_yaxis().set_major_formatter(formatter)

ax.set_xticks(
    [
        16,
        32,
        128,
        256,
        4 * 1024,
        6 * 1024,
        8 * 1024,
        20 * 1024,
        30 * 1024,
        60 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
    ]
)
ax.set_xlim([8, 800 * 1024])


ax.set_ylim([0, 980])

ax.set_yticks((0, 30, 100, 200, 300, 400, 500, 600, 700, 800, 900))

fig.autofmt_xdate()
ax.legend()
ax.set_ylim([0, ax.get_ylim()[1]])
fig.tight_layout(pad=0)
fig.savefig("latencies" + ("_" + sys.argv[1] if len(sys.argv) > 1 else "") + ".svg")
fig.savefig("latencies" + ("_" + sys.argv[1] if len(sys.argv) > 1 else "") + ".pdf")

plt.show()
