import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    path = Path(sys.argv[1])
    print('## ЛР2 — модульные тесты\n')
    if not path.exists():
        print('XML-отчёт не создан. Проверьте журнал запуска.')
        return
    root = ET.parse(path).getroot()
    print('| Тест | Статус | Время, с |\n|---|---|---:|')
    for case in root.iter('testcase'):
        status = next((x for x in ('failure', 'error', 'skipped') if case.find(x) is not None), 'passed')
        print(f"| {case.attrib['name'].replace('|', '/')} | {status} | {case.attrib.get('time', '0')} |")
    print()


if __name__ == '__main__':
    main()
