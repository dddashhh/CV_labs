import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def load(name):
    path = os.path.join(R, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_metric(rows, series, filename, title, ylabel):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for key, label, marker in series:
        if key not in rows[0]:
            continue
        points = [(int(row["pixels"]), float(row[key])) for row in rows
                  if row.get(key, "") not in ("", None)]
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    marker + "-", label=label)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число пикселей")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(R, filename), dpi=140)
    plt.close(fig)
    print(f"saved {filename}")


def plot_classes(rows, size, filename):
    sel = [r for r in rows if r["size"] == str(size)]
    if not sel:
        return
    classes = list(dict.fromkeys(r["image_class"] for r in sel))
    impls = [("native", "Python native"), ("opencv", "OpenCV"), ("pil", "Pillow")]
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (impl, label) in enumerate(impls):
        vals = []
        for c in classes:
            match = [float(r["median_ms"]) for r in sel
                     if r["image_class"] == c and r["implementation"] == impl]
            vals.append(match[0] if match else 0)
        ax.bar([k + (i - 1) * width for k in range(len(classes))], vals,
               width, label=label)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Медианное время, мс")
    ax.set_title(f"Дилатация 3×3 по классам изображений ({size}×{size})")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(R, filename), dpi=140)
    plt.close(fig)
    print(f"saved {filename}")


def main():
    os.makedirs(R, exist_ok=True)

    plot_metric(load("bench_python.csv"), [
        ("native_median_ms", "Python native", "o"),
        ("opencv_median_ms", "OpenCV", "s"),
        ("pil_median_ms", "Pillow", "^"),
    ], "benchmark.png", "Дилатация 3×3: медианное время", "Время, мс")

    plot_metric(load("bench_pipeline.csv"), [
        ("pipeline_libraries_median_ms", "Библиотеки", "s"),
        ("pipeline_native_dilation_median_ms", "Native dilation + OpenCV threshold", "o"),
        ("pipeline_fully_native_median_ms", "Полностью native", "d"),
    ], "benchmark_pipeline.png", "Бинаризация + дилатация", "Время, мс")

    plot_classes(load("bench_image_classes.csv"), 512, "benchmark_classes.png")


if __name__ == "__main__":
    main()