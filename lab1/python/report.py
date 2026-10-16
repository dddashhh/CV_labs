import csv
import os

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
IMG_DIR = os.path.join(R, "images")
PAGES_BASE = os.environ.get(
    "PAGES_BASE_URL",
    "https://dddashhh.github.io/CV_labs",
)


def load(name):
    path = os.path.join(R, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows, columns):
    if not rows:
        return "_Нет данных._\n"
    out = ["| " + " | ".join(title for title, _ in columns) + " |",
           "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(key, "")) for _, key in columns) + " |")
    return "\n".join(out) + "\n"


def print_panels():
    if not os.path.isdir(IMG_DIR):
        print("_Нет изображений._\n"); return
    panels = sorted(f for f in os.listdir(IMG_DIR) if f.endswith("_panel.png"))
    if not panels:
        print("_Нет изображений._\n"); return
    for name in panels:
        label = name[:-len("_panel.png")]
        url = f"{PAGES_BASE}/lab1/images/{name}"
        print(f"**{label}**\n")
        print(f'<img src="{url}" alt="{label}" width="800">\n')

def main():
    print("# Benchmark report — Python image processing\n")

    print("## Примеры обработки\n")
    print("Слева направо: original (gray), binary, dilated 3×3, difference.\n")
    print_panels()

    print("## Дилатация 3×3\n")
    print(markdown_table(load("bench_python.csv"), [
        ("Размер", "size"),
        ("Native median, ms", "native_median_ms"),
        ("OpenCV median, ms", "opencv_median_ms"),
        ("Pillow median, ms", "pil_median_ms"),
        ("Native/OpenCV speedup", "native_speedup_vs_opencv"),
        ("Native equal", "native_equal"),
        ("Pillow equal", "pil_equal"),
    ]))

    print("## Бинаризация + дилатация\n")
    print(markdown_table(load("bench_pipeline.csv"), [
        ("Размер", "size"),
        ("Библиотеки, ms", "pipeline_libraries_median_ms"),
        ("Fully native, ms", "pipeline_fully_native_median_ms"),
        ("Fully native speedup", "pipeline_fully_native_speedup_vs_libraries"),
        ("Fully native equal", "pipeline_fully_native_equal"),
    ]))

    classes = load("bench_image_classes.csv")
    print("## Корректность на классах изображений\n")
    if classes:
        total = len(classes)
        ok = sum(1 for r in classes if r["equal"] == "True")
        print(f"Совпало с OpenCV: **{ok} из {total}** (класс × размер × реализация).\n")
    else:
        print("_Нет данных._\n")


if __name__ == "__main__":
    main()