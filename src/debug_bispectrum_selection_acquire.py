#!/usr/bin/env python3
import json
from pathlib import Path
import bispectrum_selection_residual_validate as b
C=json.loads(Path('data/real/dr11/expanded48/provenance.json').read_text())
r=C['regions'][0]
name,d,p=b.acquire_candidate(r)
print(json.dumps({'status':'ACQUIRE_OK','field':name,'brick':d['brick'],'valid_fraction':d['valid_fraction'],'n_patches':d['n_patches'],'source_rows':d['source_rows']},indent=2))
