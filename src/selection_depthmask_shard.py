#!/usr/bin/env python3
"""Acquire one shard of REAL_DR11 depth-r + MASKBITS selection controls.

Uses provenance-verified expanded48 source CSVs from the immutable Actions
artifact. Only the official coadd depth-r and maskbits products are newly
fetched. Outputs coarsened 64x64 counts/selection arrays; no science metric is
computed in this acquisition stage.
"""
from __future__ import annotations
import argparse,gzip,hashlib,io,json,time
from pathlib import Path
import numpy as np,pandas as pd,requests
from astropy.io import fits
from astropy.wcs import WCS
from dl import queryClient as qc

GRID=64;HALF=.25;BIT_COUNT=20
BASE='https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/coadd';TABLE='ls_dr11.tractor_s'
def sha(b):return hashlib.sha256(b).hexdigest()
def q(sql,attempts=5):
 last=None
 for i in range(attempts):
  try:
   x=qc.query(sql=sql,fmt='csv',async_=False);x=x.decode() if isinstance(x,bytes) else x
   if not isinstance(x,str):raise RuntimeError(type(x))
   return x
  except Exception as e:last=e;time.sleep(2*(i+1))
 raise RuntimeError(last)
def qdf(sql):
 from io import StringIO
 return pd.read_csv(StringIO(q(sql)))
def sep2(ra,dec,ra0,dec0):
 dra=((np.asarray(ra,float)-ra0+180)%360)-180;return (dra*np.cos(np.deg2rad(dec0)))**2+(np.asarray(dec,float)-dec0)**2
def choose_brick(ra,dec):
 for rad in [.01,.02,.04,.08]:
  sql=f"SELECT brickname,ra,dec FROM {TABLE} WHERE brick_primary=1 AND q3c_radial_query(ra,dec,{ra:.8f},{dec:.8f},{rad:.8f})"
  d=qdf(sql)
  if len(d):
   i=int(np.argmin(sep2(d.ra,d.dec,ra,dec)));return str(d.iloc[i].brickname).strip(),sql,float(np.sqrt(sep2([d.iloc[i].ra],[d.iloc[i].dec],ra,dec)[0]))
 raise RuntimeError(f'no primary source near {ra},{dec}')
def get(url,attempts=4):
 last=None
 for i in range(attempts):
  try:
   r=requests.get(url,timeout=120);r.raise_for_status();b=r.content
   if not b.startswith(b'SIMPLE'):raise RuntimeError(b[:60])
   return b
  except Exception as e:last=e;time.sleep(2*(i+1))
 raise RuntimeError(f'{url}: {last}')
def image(url,wanted=None):
 b=get(url)
 with fits.open(io.BytesIO(b),memmap=False) as H:
  c=[h for h in H if h.data is not None and getattr(h.data,'ndim',0)==2]
  p=next((h for h in c if wanted and (str(h.name).upper()==wanted.upper() or str(h.header.get('EXTNAME','')).upper()==wanted.upper())),c[0])
  return np.asarray(p.data).copy(),p.header.copy(),{'url':url,'sha256':sha(b),'bytes':len(b),'shape':list(p.data.shape)}
def verify(meta,root):
 p=root/Path(meta['file']).name;gz=p.read_bytes();raw=gzip.decompress(gz)
 if sha(gz)!=meta['stored_gzip_sha256'] or sha(raw)!=meta['canonical_csv_sha256']:raise RuntimeError(f'hash mismatch {p}')
 d=pd.read_csv(p)
 if list(d.columns)!=['ra','dec'] or len(d)!=int(meta['rows']):raise RuntimeError(f'bad source {p}')
 return d
def samples(a):
 ny,nx=a.shape;o=np.array([.125,.375,.625,.875]);yy=np.clip(((np.arange(GRID)[:,None]+o)/GRID*ny).astype(int),0,ny-1);xx=np.clip(((np.arange(GRID)[:,None]+o)/GRID*nx).astype(int),0,nx-1);return a[yy[:,None,:,None],xx[None,:,None,:]]
def features(depth,mask):
 s=samples(np.asarray(depth,float));good=np.isfinite(s)&(s>0);v=np.where(good,np.log1p(np.clip(s,0,None)),np.nan)
 f=[np.nan_to_num(np.nanmean(v,(2,3)),nan=0),np.nan_to_num(np.nanstd(v,(2,3)),nan=0),good.mean((2,3))];names=['log_depth_mean','log_depth_std','depth_positive_frac']
 m=samples(np.asarray(mask,np.int64));f += [((m&1)==0).mean((2,3)),(m==0).mean((2,3))];names += ['primary_frac','clean_frac']
 for b in range(BIT_COUNT):f.append(((m&(1<<b))!=0).mean((2,3)));names.append(f'maskbit_{b}_frac')
 return np.stack(f,-1),names
def counts(df,wcs,shape):
 x,y=wcs.world_to_pixel_values(df.ra.to_numpy(float),df.dec.to_numpy(float));ny,nx=shape;k=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<nx)&(y>=0)&(y<ny);bx=np.floor(x[k]/nx*GRID).astype(int);by=np.floor(y[k]/ny*GRID).astype(int);g=np.zeros((GRID,GRID));np.add.at(g,(by,bx),1);return g,int(k.sum())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--nshards',type=int,default=12);a=ap.parse_args();root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);P=json.loads((root/'provenance.json').read_text());regs=P['regions']
 if P.get('status')!='REAL_DR11' or len(regs)!=48:raise RuntimeError('48 REAL_DR11 fields required')
 manifest=[]
 for idx in range(a.shard,len(regs),a.nshards):
  m=regs[idx];name=m['name'];ra=float(m['center_ra_deg']);dec=float(m['center_dec_deg']);src=verify(m,root);brick,bq,near=choose_brick(ra,dec);r=f'{BASE}/{brick[:3]}/{brick}';du=f'{r}/legacysurvey-{brick}-depth-r.fits.fz';mu=f'{r}/legacysurvey-{brick}-maskbits.fits.fz';print(f'[shard {a.shard}] {name}->{brick}',flush=True);d,h,dp=image(du,'DEPTH_R');mb,_,mp=image(mu,'MASKBITS');sel,names=features(d,mb);c,nin=counts(src,WCS(h),d.shape);primary=sel[:,:,names.index('primary_frac')];valid=(primary>=.9375)&(sel[:,:,names.index('depth_positive_frac')]>=.9375);c=c/np.clip(primary,.25,1);np.savez_compressed(out/f'{name}.npz',counts=c,selection=sel,valid=valid.astype(np.uint8));manifest.append({'index':idx,'field':name,'brick':brick,'fixed_center_ra_deg':ra,'fixed_center_dec_deg':dec,'nearest_primary_source_deg':near,'brick_choice_query':bq,'source_file':Path(m['file']).name,'source_rows':int(m['rows']),'sources_inside_wcs':nin,'feature_names':names,'products':{'depth_r':dp,'maskbits':mp},'valid_cell_fraction':float(valid.mean())})
 (out/f'manifest_{a.shard:02d}.json').write_text(json.dumps({'status':'REAL_DR11_DEPTHMASK_SHARD','shard':a.shard,'nshards':a.nshards,'regions':manifest},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
