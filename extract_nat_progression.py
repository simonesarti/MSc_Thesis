import os
import json
import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--start", type=int, default=1)
parser.add_argument("--stop", type=int, default=30)
args = parser.parse_args()

iter_range = range(args.start, args.stop + 1)
iterations = np.array(iter_range)

plots_save_path = os.path.join(args.path, "iter_plots")
os.makedirs(plots_save_path, exist_ok=True)

# ---------------------------------------------------------------------------------------------

# get objectives
iter_results = os.path.join(args.path, f"iter_{args.start}", "archive.json")
with open(iter_results, "r") as f:
    data = json.load(f)
objs = list(data[0].keys())
objs.remove("arch")
other_objs = objs.copy()
other_objs.remove("acc")

# prepare structure to save results
results = {}
for obj in objs:
    results[obj] = []

for i in iterations:
    iter_results = os.path.join(args.path, f"iter_{i}", "archive.json")
    with open(iter_results, "r") as f:
        data = json.load(f)

    results["acc"].append(np.array([arch["acc"] for arch in data]))
    for obj in other_objs:
        results[obj].append(np.array([arch[obj] for arch in data]))

# convert to numpy array
for obj in objs:
    results[obj] = np.array(results[obj])

# Creating figure
for obj in other_objs:
    fig = plt.figure(figsize=(17, 17))
    ax = plt.axes(projection="3d")

    colors = cm.rainbow(np.linspace(0, 1, len(iterations)))
    for iteration, c in zip(iterations, colors):
        ax.scatter3D(iteration, results[obj][iteration - args.start], results["acc"][iteration - args.start], color=c)

    ax.set_xlabel("steps")
    ax.set_xticks(iterations)
    ax.set_ylabel(obj)
    ax.set_zlabel("top1")

    plt.savefig(os.path.join(plots_save_path, f"all_3d_{obj}_{args.start}_{args.stop}.png"))
    plt.clf()
    plt.close()

# Creating plot
fig, axs = plt.subplots(len(objs), 1, sharex=True)
colors = cm.rainbow(np.linspace(0, 1, len(iterations)))
for iteration, c in zip(iterations, colors):
    for i, obj in enumerate(objs):
        if len(objs) == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.scatter(np.array([iteration] * len(results[obj][iteration - args.start])),
                       results[obj][iteration - args.start], color=c, s=50, alpha=0.5)
        ax.set_xlabel('iterations')
        ax.set_xticks(iterations)
        ax.set_ylabel(obj)

plt.savefig(os.path.join(plots_save_path, f"all_2d_sub_{args.start}_{args.stop}.png"))
plt.clf()
plt.close()

# ---------------------------------------------------------------------------------------------


# get objectives
iter_results = os.path.join(args.path, f"iter_{args.start}", "archive.json")
with open(iter_results, "r") as f:
    data = json.load(f)
objs = list(data[0].keys())
objs.remove("arch")
other_objs = objs.copy()
other_objs.remove("acc")

# prepare structure to save results
results = {}
for obj in objs:
    results[obj] = []

old_data = None
for i in iterations:

    iter_results = os.path.join(args.path, f"iter_{i}", "archive.json")
    with open(iter_results, "r") as f:
        data = json.load(f)

    if old_data is None:
        results["acc"].append(np.array([arch["acc"] for arch in data]))
        for obj in other_objs:
            results[obj].append(np.array([arch[obj] for arch in data]))

    else:

        new_indexes = []
        for i, arch in enumerate(data):
            found = False
            for old_arch in old_data:
                if arch["arch"] == old_arch["arch"]:
                    found = True
            if not found:
                new_indexes.append(i)

        acc_append = []
        for idx in new_indexes:
            acc_append.append(data[idx]["acc"])
        results["acc"].append(np.array(acc_append))

        for obj in other_objs:
            obj_append = []
            for idx in new_indexes:
                obj_append.append(data[idx][obj])
            results[obj].append(np.array(obj_append))

        # print(new_indexes)
    old_data = data

# convert to numpy array
for obj in objs:
    results[obj] = np.array(results[obj])

# Creating figure
for obj in other_objs:
    fig = plt.figure(figsize=(17, 17))
    ax = plt.axes(projection="3d")

    colors = cm.rainbow(np.linspace(0, 1, len(iterations)))
    for iteration, c in zip(iterations, colors):
        ax.scatter3D(iteration, results[obj][iteration - args.start], results["acc"][iteration - args.start], color=c)

    ax.set_xlabel("steps")
    ax.set_xticks(iterations)
    ax.set_ylabel(obj)
    ax.set_zlabel("top1")

    plt.savefig(os.path.join(plots_save_path, f"update_3d_{obj}_{args.start}_{args.stop}.png"))
    plt.clf()
    plt.close()

# Creating plot
fig, axs = plt.subplots(len(objs), 1, sharex=True, squeeze=False)

colors = cm.rainbow(np.linspace(0, 1, len(iterations)))
for iteration, c in zip(iterations, colors):
    for i, obj in enumerate(objs):
        if len(objs) == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.scatter(np.array([iteration] * len(results[obj][iteration - args.start])),
                       results[obj][iteration - args.start], color=c, s=50, alpha=0.5)
        ax.set_xlabel('iterations')
        ax.set_xticks(iterations)
        ax.set_ylabel(obj)

plt.savefig(os.path.join(plots_save_path, f"update_2d_sub_{args.start}_{args.stop}.png"))
plt.clf()
plt.close()

# Creating figure
for obj in other_objs:

    colors = cm.rainbow(np.linspace(0, 1, len(iterations)))
    for iteration, c in zip(iterations, colors):
        plt.scatter(results[obj][iteration - args.start], results["acc"][iteration - args.start], color=c, s=30,
                    alpha=0.7, label=str(iteration))

    plt.xlabel(obj)
    plt.ylabel("top1")
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(plots_save_path, f"update_2d_overlapped_{obj}_{args.start}_{args.stop}.png"))
    plt.clf()
    plt.close()
