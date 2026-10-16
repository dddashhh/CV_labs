import csv
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc

import cv2
import numpy as np
import PIL
from PIL import Image

from dilation import binarize, binarize_native, dilate_native, dilate_opencv, dilate_pil
from make_test_image import make, make_case

SIZES = [64, 128, 256, 512, 1024]
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
LIB_REPEATS = 7
NATIVE_REPEATS = 3


def measure(fn, arg, repeats):
    """Return robust timing statistics in milliseconds."""
    fn(arg)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn(arg)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    median = statistics.median(samples)
    return {
        "min_ms": min(samples),
        "median_ms": median,
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p95_ms": float(np.percentile(samples, 95)),
        "max_ms": max(samples),
        "samples": len(samples),
    }


def memory_peak_bytes(fn, arg):
    """Measure Python-tracked peak allocations."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        fn(arg)
        _, peak = tracemalloc.get_traced_memory()
        return int(peak)
    finally:
        tracemalloc.stop()


def correctness(actual, expected):
    a = np.asarray(actual)
    b = np.asarray(expected)
    if a.shape != b.shape:
        return {
            "equal": False, "mismatch_pixels": int(max(a.size, b.size)),
            "mismatch_pct": 100.0, "max_abs_error": -1,
            "mean_abs_error": -1.0,
        }
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mismatches = int(np.count_nonzero(diff))
    total = int(diff.size)
    return {
        "equal": mismatches == 0,
        "mismatch_pixels": mismatches,
        "mismatch_pct": 100.0 * mismatches / total if total else 0.0,
        "max_abs_error": int(diff.max()) if total else 0,
        "mean_abs_error": float(diff.mean()) if total else 0.0,
    }


def rounded_metrics(metrics):
    return {k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in metrics.items()}


def performance_fields(prefix, fn, image, repeats):
    m = measure(fn, image, repeats)
    pixels = int(image.shape[0] * image.shape[1])
    return {
        f"{prefix}_min_ms": m["min_ms"],
        f"{prefix}_median_ms": m["median_ms"],
        f"{prefix}_mean_ms": m["mean_ms"],
        f"{prefix}_stdev_ms": m["stdev_ms"],
        f"{prefix}_p95_ms": m["p95_ms"],
        f"{prefix}_max_ms": m["max_ms"],
        f"{prefix}_samples": m["samples"],
        f"{prefix}_mpixels_per_sec": pixels / (m["median_ms"] * 1000),
        f"{prefix}_peak_python_bytes": (memory_peak_bytes(fn, image) if pixels <= 65536 else ""),
    }


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)



GENERATED_CLASSES = [
    "geometry", "checkerboard", "sparse", "uniform_black", "uniform_white",
]
LOADED_CLASSES = [
    "dog_photo_gallai", "dog_photo_bluz", "dog_mask",
]


def foreground_pct(binary):
    """The proportion of white pixels, %."""
    return 100.0 * np.count_nonzero(binary) / binary.size


def benchmark_image_classes():
    rows, failures = [], []
    sources = [(name, "generated") for name in GENERATED_CLASSES] + \
              [(name, "loaded") for name in LOADED_CLASSES]

    for n in SIZES:
        native_repeats = 1 if n >= 1024 else (2 if n >= 512 else NATIVE_REPEATS)
        for name, source in sources:
            try:
                gray = make_case(name, n)
            except FileNotFoundError as exc:
                failures.append(f"{name} {n}x{n}: {exc}")
                print(f"MISSING {name}: {exc}", file=sys.stderr)
                continue
            binary = binarize(gray)
            reference = dilate_opencv(binary)
            fg_before = foreground_pct(binary)
            fg_after = foreground_pct(reference)
            for impl, fn, repeats in [
                ("native", dilate_native, native_repeats),
                ("opencv", dilate_opencv, LIB_REPEATS),
                ("pil", dilate_pil, LIB_REPEATS),
            ]:
                check = correctness(fn(binary), reference)
                metrics = measure(fn, binary, repeats)
                pixels = n * n
                rows.append({
                    "image_class": name, "source": source,
                    "size": n, "pixels": pixels, "implementation": impl,
                    "foreground_before_pct": round(fg_before, 4),
                    "foreground_after_pct": round(fg_after, 4),
                    "equal": check["equal"],
                    "mismatch_pixels": check["mismatch_pixels"],
                    "mismatch_pct": round(check["mismatch_pct"], 8),
                    "median_ms": round(metrics["median_ms"], 6),
                    "mean_ms": round(metrics["mean_ms"], 6),
                    "p95_ms": round(metrics["p95_ms"], 6),
                    "stdev_ms": round(metrics["stdev_ms"], 6),
                    "samples": metrics["samples"],
                    "mpixels_per_sec": round(pixels / (metrics["median_ms"] * 1000), 6),
                    "peak_python_bytes": memory_peak_bytes(fn, binary) if pixels <= 65536 else "",
                })
                if not check["equal"]:
                    failures.append(f"{name} {n}x{n}: {impl} differs from OpenCV: {check}")
            print(f"case={name:18s} source={source:9s} size={n:4d}x{n:<4d} "
                  f"fg={fg_before:5.1f}%->{fg_after:5.1f}%")
    write_csv(os.path.join(OUT, "bench_image_classes.csv"), rows)
    return rows, failures


def labeled(img, text):
    bar = np.full((28, img.shape[1]), 255, dtype=np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def save_example_images(size=512):
    out_dir = os.path.join(OUT, "images")
    os.makedirs(out_dir, exist_ok=True)
    cases = []
    test_path = os.path.join(os.path.dirname(__file__), "..", "images", "test.png")
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if test is not None:
        cases.append(("test_png", cv2.resize(test, (size, size))))
    for name in GENERATED_CLASSES + LOADED_CLASSES:
        cases.append((name, make_case(name, size)))

    for name, gray in cases:
        binary = binarize(gray)
        dilated = dilate_native(binary)
        diff = cv2.absdiff(dilated, binary)
        panel = np.hstack([
            labeled(gray, "original (gray)"), labeled(binary, "binary"),
            labeled(dilated, "dilated 3x3"), labeled(diff, "difference"),
        ])
        cv2.imwrite(os.path.join(out_dir, f"{name}_panel.png"), panel)
        cv2.imwrite(os.path.join(out_dir, f"{name}_dilated.png"), dilated)
        print(f"saved results/images/{name}_panel.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    dilation_rows, pipeline_rows = [], []
    failures = []

    for n in SIZES:
        gray = make(n)
        binary = binarize(gray)
        reference = dilate_opencv(binary)
        native_repeats = 1 if n >= 1024 else (2 if n >= 512 else 5)

        implementations = [
            ("native", dilate_native, native_repeats),
            ("opencv", dilate_opencv, LIB_REPEATS),
            ("pil", dilate_pil, LIB_REPEATS),
        ]
        row = {"size": n, "pixels": n*n}
        for name, fn, repeats in implementations:
            result = fn(binary)
            check = correctness(result, reference)
            row[f"{name}_equal"] = check["equal"]
            row[f"{name}_mismatch_pixels"] = check["mismatch_pixels"]
            row[f"{name}_mismatch_pct"] = round(check["mismatch_pct"], 8)
            row[f"{name}_max_abs_error"] = check["max_abs_error"]
            row[f"{name}_mean_abs_error"] = round(check["mean_abs_error"], 8)
            row.update(performance_fields(name, fn, binary, repeats))
            if not check["equal"]:
                failures.append(f"{n}x{n}: {name} differs from OpenCV: {check}")

        for name in ("native", "pil"):
            denom = row["opencv_median_ms"]
            row[f"{name}_speedup_vs_opencv"] = (
                round(denom / row[f"{name}_median_ms"], 6)
                if row[f"{name}_median_ms"] else ""
            )

        binary_native = binarize_native(gray)
        bin_check = correctness(binary_native, binary)
        lib_pipe = lambda image: dilate_opencv(binarize(image))
        native_dilate_pipe = lambda image: dilate_native(binarize(image))
        native_full_pipe = lambda image: dilate_native(binarize_native(image))
        pipe_reference = lib_pipe(gray)
        pipe_row = {
            "size": n, "pixels": n*n,
            "binarize_equal": bin_check["equal"],
            "binarize_mismatch_pixels": bin_check["mismatch_pixels"],
            "binarize_mismatch_pct": round(bin_check["mismatch_pct"], 8),
        }
        for name, fn, repeats in [
            ("binarize_opencv", binarize, LIB_REPEATS),
            ("binarize_native", binarize_native, native_repeats),
            ("pipeline_libraries", lib_pipe, LIB_REPEATS),
            ("pipeline_native_dilation", native_dilate_pipe, native_repeats),
            ("pipeline_fully_native", native_full_pipe, native_repeats),
        ]:
            pipe_row.update(performance_fields(name, fn, gray, repeats))
            if name.startswith("pipeline_"):
                check = correctness(fn(gray), pipe_reference)
                pipe_row[f"{name}_equal"] = check["equal"]
                pipe_row[f"{name}_mismatch_pixels"] = check["mismatch_pixels"]
                pipe_row[f"{name}_mismatch_pct"] = round(check["mismatch_pct"], 8)
                pipe_row[f"{name}_max_abs_error"] = check["max_abs_error"]
                if not check["equal"]:
                    failures.append(f"{n}x{n}: {name} differs from library pipeline")
        for name in ("pipeline_native_dilation", "pipeline_fully_native"):
            denom = pipe_row["pipeline_libraries_median_ms"]
            pipe_row[f"{name}_speedup_vs_libraries"] = (
                round(denom / pipe_row[f"{name}_median_ms"], 6)
                if pipe_row[f"{name}_median_ms"] else ""
            )
        if not bin_check["equal"]:
            failures.append(f"{n}x{n}: native binarization differs from OpenCV")

        dilation_rows.append(rounded_metrics(row))
        pipeline_rows.append(rounded_metrics(pipe_row))
        print(
            f"{n:4}x{n:<4} px={n*n:>8} | "
            f"dilate median ms: native={row['native_median_ms']:.4f}, "
            f"OpenCV={row['opencv_median_ms']:.4f}, PIL={row['pil_median_ms']:.4f} | "
            f"speedup native/OpenCV={row['native_speedup_vs_opencv']:.3f}x"
        )

    write_csv(os.path.join(OUT, "bench_python.csv"), dilation_rows)
    write_csv(os.path.join(OUT, "bench_pipeline.csv"), pipeline_rows)

    _, class_failures = benchmark_image_classes()
    failures.extend(class_failures)
    save_example_images()
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "sizes": SIZES,
        "generated_classes": GENERATED_CLASSES,
        "loaded_classes": LOADED_CLASSES,
        "library_repeats": LIB_REPEATS,
        "native_repeats": NATIVE_REPEATS,
        "timing_clock": "time.perf_counter_ns",
        "memory_metric": "tracemalloc peak Python-tracked bytes; excludes native allocations",
    }
    with open(os.path.join(OUT, "benchmark_environment.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)

    gray = make(512)
    binary = binarize(gray)
    dilated = dilate_opencv(binary)
    cv2.imwrite(os.path.join(OUT, "example_gray.png"), gray)
    cv2.imwrite(os.path.join(OUT, "example_binary.png"), binary)
    cv2.imwrite(os.path.join(OUT, "example_dilated.png"), dilated)
    cv2.imwrite(os.path.join(OUT, "example_diff.png"), cv2.absdiff(dilated, binary))
    cv2.imwrite(os.path.join(OUT, "example_panel.png"),
                np.hstack([gray, binary, dilated, cv2.absdiff(dilated, binary)]))

    if failures:
        print("BENCHMARK FAILED: correctness checks failed:", file=sys.stderr)
        for failure in failures:
            print(f" - {failure}", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll correctness checks passed. CSV and environment metadata written to {OUT}")


if __name__ == "__main__":
    main()
