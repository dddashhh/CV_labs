import argparse
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
REPO = 'https://github.com/zsyOAOA/ResShift.git'
RELEASE = 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/'
FILES = ('resshift_realsrx4_s15_v2.pth', 'autoencoder_vq_f4.pth')


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        partial = path.with_suffix(path.suffix + '.part')
        urlretrieve(url, partial)
        partial.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', default='bb03b7d21614cace01787e097c8a6ab6b945227d')
    args = parser.parse_args()
    repo = ROOT / 'ResShift'
    if not repo.exists():
        subprocess.run(['git', 'clone', '--branch', 'journal', REPO, str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', args.revision], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    (ROOT / 'weights').mkdir(exist_ok=True)
    (ROOT / 'weights/revision.txt').write_text(revision + '\n')
    for name in FILES:
        print(download(RELEASE + name, ROOT / 'weights' / name))
    print('ResShift revision:', revision)


if __name__ == '__main__':
    main()
