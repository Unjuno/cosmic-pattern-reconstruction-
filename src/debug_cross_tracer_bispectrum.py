#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import tracer_split_validate as tr
import tracer_split_native_patch as native
import bispectrum_phase_validate as bp
native.install()
C=json.loads(Path('data/real/dr11/expanded48/provenance.json').read_text()); r=C['regions'][0]
brick,bq=tr.choose_brick(float(r['center_ra_deg']),float(r['center_dec_deg']))
print('brick',brick,flush=True)
d,p=tr.get_tractor(brick); print('rows',len(d),'types',d.type.value_counts().to_dict(),flush=True)
psf=d[d.type.eq('PSF')].copy(); ext=d[d.type.isin(tr.EXT_TYPES)].copy(); n=min(len(psf),len(ext)); print('psf/ext/n',len(psf),len(ext),n,flush=True)
rng=np.random.default_rng(20260831); psf_eq=psf.iloc[rng.choice(len(psf),n,replace=False)]; ext_eq=ext.iloc[rng.choice(len(ext),n,replace=False)]
gp=tr.grid_from(psf_eq); ge=tr.grid_from(ext_eq); print('grid sums',gp.sum(),ge.sum(),flush=True)
print('gaussian std',bp.gaussianize(gp).std(),bp.gaussianize(ge).std(),flush=True)
