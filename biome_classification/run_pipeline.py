"""Run all 325 internal classification tasks and consolidate their test metrics."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def commands(device):
    train = 'model_training/'
    jobs = [[train+'train.py', '--encoding', 'hilbert', '--prepare-only']]
    jobs += [[train+'train.py', '--encoding', encoding, '--device', device]
             for encoding in ('hilbert', 'rowwise', '1d')]
    jobs += [[train+'phylogeny_baselines/train.py', '--method', method, '--device', device]
             for method in ('deepphylo_taxonomy', 'popphy_cnn')]
    jobs += [[train+'classical_order_control/train.py', '--model', method]
             for method in ('random_forest', 'logistic_regression')]
    jobs += [[train+'results/summarize_training.py']]
    return [[sys.executable, *job] for job in jobs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without reading data or training')
    args = parser.parse_args()
    for command in commands(args.device):
        print(' '.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=HERE, check=True)


if __name__ == '__main__':
    main()
