import argparse
import csv
import html
import math
import xml.etree.ElementTree as ET
from pathlib import Path


COLUMNS = (
    ('Деградация', 'degradation'), ('Маска', 'mask'), ('N', 'n'),
    ('PSNR, дБ', 'psnr_db'), ('LPIPS', 'lpips'), ('CLIP-IQA', 'clip_iqa'),
    ('PSNR ROI, дБ', 'psnr_roi_db'), ('Ошибка вне маски', 'outside_max_error'),
    ('Время, с', 'seconds'),
)
ROI_COLUMNS = (
    ('Изображение', 'image'), ('Деградация', 'degradation'),
    ('Ones, дБ', 'ones_roi_db'), ('Mixed, дБ', 'mixed_roi_db'),
    ('Δ PSNR ROI, дБ', 'gain_roi_db'),
)


def load(path):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def fmt(value):
    if value is None or value == '':
        return '—'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return '—'
    if math.isinf(number):
        return '∞' if number > 0 else '−∞'
    if isinstance(value, str) and value.isdigit():
        return value
    return f'{number:.3f}'


def table(rows, columns, markup='md'):
    if not rows:
        return 'Нет данных.' if markup == 'md' else '<p>Нет данных.</p>'
    headers = [title for title, _ in columns]
    cells = [[fmt(row.get(key)) for _, key in columns] for row in rows]
    if markup == 'html':
        head = '<tr>' + ''.join(f'<th>{html.escape(x)}</th>' for x in headers) + '</tr>'
        body = ''.join('<tr>' + ''.join(f'<td>{html.escape(x)}</td>' for x in row) + '</tr>' for row in cells)
        return f'<table><thead>{head}</thead><tbody>{body}</tbody></table>'
    safe = lambda text: text.replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(safe, row)) + ' |' for row in cells])


def junit_report(path):
    if not path.is_file():
        return '## ЛР3 — тестирование\n\nJUnit-отчёт не создан. Проверьте логи установки и запуска тестов.\n'
    cases = list(ET.parse(path).getroot().iter('testcase'))
    counts = dict.fromkeys(('passed', 'failed', 'error', 'skipped'), 0)
    rows = []
    for case in cases:
        status = next((kind for kind, tag in [('failed', 'failure'), ('error', 'error'), ('skipped', 'skipped')]
                       if case.find(tag) is not None), 'passed')
        counts[status] += 1
        rows.append({'name': case.get('name'), 'status': status, 'time': case.get('time')})
    line = ', '.join(f'{value} {key}' for key, value in counts.items())
    return ('## ЛР3 — тестирование\n\n**' + line + '**\n\n' +
            table(rows, (('Тест', 'name'), ('Статус', 'status'), ('Время, с', 'time'))) + '\n')


def experiment_report(out):
    summary = load(out / 'summary.csv')
    pairs = load(out / 'roi_comparison.csv')
    results = load(out / 'metrics.csv')
    if not summary:
        text = '## ЛР3 — SR-inpainting\n\nПолный отчёт не сформирован. Проверьте логи эксперимента.\n'
        if results:
            text += f'\nУспешно записано конфигураций: **{len(results)}**. Частичные результаты:\n\n'
            text += table(results, (('Изображение', 'image'),) + tuple(c for c in COLUMNS if c[1] != 'n')) + '\n'
        return text
    return '\n\n'.join([
        '## ЛР3 — SR-inpainting', f'Выполнено конфигураций: **{len(results)}**.',
        'Маски: zeros — эталон HR; ones — восстановление по LR; mixed — восстановление с HR-контекстом.',
        '### Средние метрики', table(summary, COLUMNS),
        '∞ означает точное совпадение с HR; — означает неприменимую или отсутствующую метрику.',
        '### Сравнение на одинаковой ROI', table(pairs, ROI_COLUMNS),
        'Положительный Δ PSNR ROI означает преимущество HR-контекста по PSNR на восстанавливаемой области.',
        'Сравнительные панели, отдельные изображения и графики доступны в Artifacts.'],
    ) + '\n'


def write_html(out):
    summary = load(out / 'summary.csv')
    pairs = load(out / 'roi_comparison.csv')
    results = load(out / 'metrics.csv')
    metadata = out / 'run.json'
    figures = sorted((out / 'figures').glob('*.png'))
    sections = [
        '<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>ЛР3 — SR-inpainting</title>',
        '<style>body{font:16px system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#172033}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd4de;padding:7px;text-align:left}th{background:#edf2f7}img{max-width:100%;height:auto}figure{margin:24px 0}pre{white-space:pre-wrap}</style></head><body>',
        '<h1>ЛР3 — SR-inpainting</h1>', f'<p>Конфигураций: {len(results)}. Источник данных: DIV2K (NTIRE 2017).</p>',
        '<h2>Средние метрики</h2>', table(summary, COLUMNS, 'html'),
        '<h2>Сравнение на одинаковой ROI</h2>', table(pairs, ROI_COLUMNS, 'html'),
        '<p>Положительный Δ означает преимущество HR-контекста. Нулевая маска возвращает HR и не оценивает качество восстановления.</p>',
        '<h2>Графики и изображения</h2>',
    ]
    for figure in figures:
        name = html.escape(figure.name)
        sections.append(f'<figure><img src="figures/{name}" alt="{name}"><figcaption>{name}</figcaption></figure>')
    sections += ['<h2>Параметры запуска</h2>',
                 '<pre>' + html.escape(metadata.read_text(encoding='utf-8') if metadata.exists() else 'Нет данных') + '</pre>',
                 '</body></html>']
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.html').write_text('\n'.join(sections), encoding='utf-8')


def write_markdown(path, content, figures=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    links = [f"![{figure.stem}](figures/{figure.name})" for figure in figures]
    if links:
        content += "\n### Графики и изображения\n\n" + "\n\n".join(links) + "\n"
    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('tests', 'experiments'))
    parser.add_argument('path', type=Path)
    parser.add_argument('--html', action='store_true')
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    if args.mode == 'tests':
        content = junit_report(args.path)
        print(content)
        if args.markdown:
            write_markdown(args.markdown, content)
    else:
        content = experiment_report(args.path)
        print(content)
        if args.markdown:
            write_markdown(args.markdown, content, sorted((args.path / 'figures').glob('*.png')))
        if args.html and (args.path / 'summary.csv').exists():
            write_html(args.path)


if __name__ == '__main__':
    main()
