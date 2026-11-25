import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generate_report(out, image_base=None, panels=True):
    out = Path(out)
    with (out / 'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    cases = list(dict.fromkeys(r['case'] for r in rows))
    lookup = {(r['case'], r['mode']): r for r in rows}
    for key, name, factor in [('coverage', 'Coverage, %', 100), ('fps', 'Algorithm speed, frames/s', 1)]:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(cases))
        for j, mode in enumerate(('sift', 'hybrid')):
            ax.bar(x + (j-.5)*.35, [factor*float(lookup[c, mode][key]) for c in cases], width=.35, label=mode)
        ax.set_xticks(x, cases, rotation=15, ha='right')
        ax.set_ylabel(name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{key}.png', dpi=140)
        plt.close(fig)
    lines = ['# ЛР2 — трекинг плоского объекта', '', '| Видео | Метод | Кадры с рамкой (без первого) | Покрытие | FPS | Инлаеры, среднее | Ошибка репроекции, px |', '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        error = f"{float(r['mean_reprojection']):.2f}" if r['mean_reprojection'] else '—'
        lines.append(f"| {r['case']} | {r['mode']} | {r['found']}/{r['evaluated']} | {float(r['coverage']):.2%} | {float(r['fps']):.2f} | {float(r['mean_inliers']):.1f} | {error} |")
    lines += ['', 'Покрытие — доля кадров с принятой рамкой, не точность локализации. FPS учитывает вычисления трекера, без записи и декодирования видео. Без покадровой разметки нельзя измерить точность границ или долю ложных срабатываний.']
    lines += ['', '## Графики', '']
    prefix = image_base.rstrip('/') + '/' if image_base else ''
    for name in ('coverage.png', 'fps.png'):
        if (out / name).exists():
            lines += [f'![{name}]({prefix}{name})', '']
    if panels:
        lines += ['## Примеры кадров', '']
        for case in cases:
            lines += [f'### {case}', '', f'![{case}, hybrid]({prefix}{case}/hybrid/panel.jpg)', '']
    if not image_base:
        lines += ['## Видео', '']
        for case in cases:
            lines.append(f'- {case}: [SIFT]({case}/sift/tracked.mp4), [Hybrid]({case}/hybrid/tracked.mp4)')
    references = {'mona-lisa', 'mona-lisa-blur', 'mona-lisa-blur-extra-credit'}
    additional = [c for c in cases if c not in references]
    if additional:
        lines += ['', 'Дополнительные ролики: ' + ', '.join(additional) + '. Происхождение роликов определяется автором, а не именем файла.']
    else:
        lines += ['', 'Дополнительные ролики в данном запуске отсутствуют.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('lab2/results'))
    parser.add_argument('--image-base')
    parser.add_argument('--no-panels', action='store_true')
    args = parser.parse_args()
    report = generate_report(args.out, args.image_base, not args.no_panels)
    if not args.image_base:
        (args.out / 'report.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
