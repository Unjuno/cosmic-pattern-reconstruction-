#!/usr/bin/env python3
"""Availability-gated wrapper for sharded REAL_DR11 depth+MASKBITS acquisition."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from astropy.wcs import WCS
import selection_depthmask_shard as c

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--nshards',type=int,default=12);a=ap.parse_args();root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);P=json.loads((root/'provenance.json').read_text());regs=P['regions'];rows=[]
 if P.get('status')!='REAL_DR11' or len(regs)!=48:raise RuntimeError('48 REAL_DR11 fields required')
 for idx in range(a.shard,len(regs),a.nshards):
  m=regs[idx];name=m['name'];ra=float(m['center_ra_deg']);dec=float(m['center_dec_deg'])
  try:
   src=c.verify(m,root);brick,bq,near=c.choose_brick(ra,dec);r=f'{c.BASE}/{brick[:3]}/{brick}';du=f'{r}/legacysurvey-{brick}-depth-r.fits.fz';mu=f'{r}/legacysurvey-{brick}-maskbits.fits.fz';print(f'[shard {a.shard}] {name}->{brick}',flush=True);d,h,dp=c.image(du,'DEPTH_R');mb,_,mp=c.image(mu,'MASKBITS')
   if d.shape!=mb.shape:raise RuntimeError('depth/mask shape mismatch')
   sel,names=c.features(d,mb);cnt,nin=c.counts(src,WCS(h),d.shape);primary=sel[:,:,names.index('primary_frac')];valid=(primary>=.9375)&(sel[:,:,names.index('depth_positive_frac')]>=.9375);cnt=cnt/np.clip(primary,.25,1);np.savez_compressed(out/f'{name}.npz',counts=cnt,selection=sel,valid=valid.astype(np.uint8));rows.append({'status':'accepted','index':idx,'field':name,'brick':brick,'fixed_center_ra_deg':ra,'fixed_center_dec_deg':dec,'nearest_primary_source_deg':near,'brick_choice_query':bq,'source_file':Path(m['file']).name,'source_rows':int(m['rows']),'sources_inside_wcs':nin,'feature_names':names,'products':{'depth_r':dp,'maskbits':mp},'valid_cell_fraction':float(valid.mean())})
  except Exception as e:
   print(f'[shard {a.shard}] REJECT {name}: {e}',flush=True);rows.append({'status':'availability_reject','index':idx,'field':name,'fixed_center_ra_deg':ra,'fixed_center_dec_deg':dec,'reason':str(e)})
 (out/f'manifest_{a.shard:02d}.json').write_text(json.dumps({'status':'REAL_DR11_DEPTHMASK_SHARD_AVAILABILITY_GATED','shard':a.shard,'nshards':a.nshards,'regions':rows},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
