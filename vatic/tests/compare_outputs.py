"""Checks if the outputs of different Vatic simulation runs are the same."""

import argparse
import bz2
import dill as pickle
from typing import Dict
import pandas as pd
from itertools import combinations as combns


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("out_fls", nargs='+')
    args = parser.parse_args()
    outputs = [dict() for _ in args.out_fls]

    for i, out_fl in enumerate(args.out_fls):
        with bz2.BZ2File(out_fl, 'r') as f:
            outputs[i] : Dict[str, pd.DataFrame] = pickle.load(f)

    assert len({tuple(out.keys()) for out in outputs}) == 1, (
        "These output files have differing levels of detail!")

    for k in outputs[0]:
        for out1, out2 in combns(outputs, 2):
            if k != 'runtimes' and isinstance(out1[k], pd.DataFrame):
                out_cmp = out1[k].compare(out2[k])

                assert out_cmp.shape == (0, 0), (
                    f"These outputs have {out_cmp} "
                    f"differing values for field `{k}`!"
                    )


if __name__ == '__main__':
    main()
