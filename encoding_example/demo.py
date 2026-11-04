"""Generate two examples, encode them, and check value preservation."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dbfile', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    abundance = np.array([[1., 2., 3.], [4., 5., 6.]])
    np.save(here / 'tax_ids.npy', np.array([562, 1280, 287], dtype=np.int64))
    np.save(here / 'abundance.npy', abundance)
    command = [sys.executable, str(here / 'encode.py'), '--tax-ids', str(here / 'tax_ids.npy'),
               '--abundance', str(here / 'abundance.npy'), '--output', str(here / 'encoded.npy')]
    if args.dbfile:
        command += ['--dbfile', str(args.dbfile.resolve())]
    subprocess.run(command, check=True)
    encoded = np.load(here / 'encoded.npy', allow_pickle=False)
    assert encoded.shape == (2, 2, 2), encoded.shape
    expected = np.concatenate([abundance, np.zeros((2, 1))], axis=1)
    np.testing.assert_array_equal(np.sort(encoded.reshape(2, 4), axis=1), np.sort(expected, axis=1))
    print('Passed: shape (2, 2, 2), all abundances preserved, one zero added per sample.')


if __name__ == '__main__':
    main()
