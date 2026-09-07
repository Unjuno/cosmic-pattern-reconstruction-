#!/usr/bin/env python3
"""Higher-density 0%-4% official DR11 point-random stress test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import point_random_selection_residual_validate as base

FRACTION = 0.04


def main() -> int:
    # Keep all science logic in the preregistered base pipeline; only increase
    # the deterministic randomized-prefix density from 2% to 4% per file.
    if '--fraction' not in sys.argv:
        sys.argv.extend(['--fraction', str(FRACTION)])
    rc = base.main()
    out = Path('results/real_dr11/point_random_selection_residual36_dense4')
    if '--out' in sys.argv:
        out = Path(sys.argv[sys.argv.index('--out') + 1])
    p = out / 'summary.json'
    if p.exists():
        d = json.loads(p.read_text())
        d['status'] = 'REAL_DR11_POINT_RANDOM_SELECTION_RESIDUAL36_DENSE4'
        d['T'] = 'same 36 fields and same pipeline; first 4% of each of 20 randomized official point-random files; k=4 IDW selection surfaces; 6-fold whole-field CV'
        d['density_role'] = 'higher point-random density stress test of finite-sampling / kNN smoothing sensitivity'
        p.write_text(json.dumps(d, indent=2, sort_keys=True) + '\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
