#!/usr/bin/env python3
"""REAL_DR11 higher-order Fourier phase-coupling / bicoherence test.

Each 64x64 DR11 count field is rank-Gaussianized to remove the one-point
marginal. We then compare closure-phase locking and amplitude-weighted
bicoherence against a paired control with the *exact same full 2D Fourier
amplitude* and randomized Hermitian phase.

The primary test pools all non-degenerate triangles with 1 <= |k| < 16.
Secondary low-low, squeezed, mid-mid and high-high triangle families are
reported descriptively. No simulated cosmology is used.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata,norm,binomtest,wilcoxon

GRID=64;HALF=.25

def tra(a,a0):return ((np.asarray(a,float)-a0+180)%360)-180

def grid(m,root):
 d=pd.read_csv(root/Path(m['file']).name);x=tra(d.ra,m['center_ra_deg']);y=d.dec-m['center_dec_deg'];k=(abs(x)<HALF)&(abs(y)<HALF)
 h,_,_=np.histogram2d(y[k],x[k],bins=GRID,range=[[-HALF,HALF],[-HALF,HALF]])
 z=np.log1p(h);med=np.median(z);s=np.median(abs(z-med))*1.4826;s=s if s>1e-6 else np.std(z)
 return (z-med)/(s if s>1e-6 else 1)

def gaussianize(z):
 r=rankdata(z.ravel(),method='average');u=(r-.5)/len(r);return norm.ppf(np.clip(u,1e-6,1-1e-6)).reshape(z.shape)

def exact_phase(g,seed):
 rng=np.random.default_rng(seed);n=rng.normal(size=g.shape);N=np.fft.fft2(n);ph=N/np.where(abs(N)>0,abs(N),1)
 amp=abs(np.fft.fft2(g));s=np.fft.ifft2(amp*ph).real;s+=g.mean()-s.mean();return s

def fft_idx(k):return int(k)%GRID

def build_triangles(seed=20260901,max_pairs=16000):
 # Restrict input modes to |component|<=15 so k1+k2 does not alias around the FFT box.
 modes=[]
 for ky in range(-15,16):
  for kx in range(-15,16):
   r=np.hypot(kx,ky)
   if 1<=r<16:modes.append((kx,ky,r))
 fam={'all':[],'low_low':[],'squeezed':[],'mid_mid':[],'high_high':[]}
 for i,(x1,y1,r1) in enumerate(modes):
  for x2,y2,r2 in modes[i:]:
   x3=x1+x2;y3=y1+y2;r3=np.hypot(x3,y3)
   if (x3==0 and y3==0) or abs(x3)>31 or abs(y3)>31 or not (1<=r3<24):continue
   t=(x1,y1,x2,y2,x3,y3)
   fam['all'].append(t)
   if r1<4 and r2<4:fam['low_low'].append(t)
   if (r1<4 and 9<=r2<16) or (r2<4 and 9<=r1<16):fam['squeezed'].append(t)
   if 4<=r1<9 and 4<=r2<9:fam['mid_mid'].append(t)
   if 9<=r1<16 and 9<=r2<16:fam['high_high'].append(t)
 rng=np.random.default_rng(seed)
 out={}
 for name,arr in fam.items():
  a=np.asarray(arr,dtype=np.int16)
  if len(a)>max_pairs:a=a[rng.choice(len(a),max_pairs,replace=False)]
  out[name]=a
 return out

def stats_for_field(g,tri):
 F=np.fft.fft2(g)
 out={}
 for name,T in tri.items():
  if len(T)==0:continue
  x1,y1,x2,y2,x3,y3=[T[:,j] for j in range(6)]
  a=F[np.mod(y1,GRID),np.mod(x1,GRID)];b=F[np.mod(y2,GRID),np.mod(x2,GRID)];c=F[np.mod(y3,GRID),np.mod(x3,GRID)]
  closure=a*b*np.conj(c)
  unit=closure/np.where(abs(closure)>0,abs(closure),1)
  phase_lock=float(abs(np.mean(unit)))
  denom=np.sqrt(np.mean(abs(a*b)**2)*np.mean(abs(c)**2))+1e-30
  bico=float(abs(np.mean(closure))/denom)
  signed=float(np.real(np.mean(closure))/denom)
  out[name]={'phase_lock':phase_lock,'bicoherence':bico,'signed_bicoherence':signed,'n_triangles':int(len(T))}
 return out

def paired(d):
 d=np.asarray(d,float);d=d[np.isfinite(d)];n=len(d);pos=int((d>0).sum())
 try:w=float(wilcoxon(d,alternative='greater').pvalue)
 except Exception:w=float('nan')
 return {'n':n,'positive':pos,'median':float(np.median(d)),'mean':float(np.mean(d)),'sign_p_one_sided':float(binomtest(pos,n,.5,alternative='greater').pvalue),'wilcoxon_p_one_sided':w}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 P=json.loads((root/'provenance.json').read_text())
 if P.get('status')!='REAL_DR11' or len(P.get('regions',[]))!=48:raise RuntimeError('48-field REAL_DR11 artifact required')
 tri=build_triangles();rows=[]
 for i,m in enumerate(P['regions']):
  g=gaussianize(grid(m,root));p=exact_phase(g,20260901+i);sr=stats_for_field(g,tri);sp=stats_for_field(p,tri)
  for fam in tri:
   for sample,s in [('real',sr[fam]),('exact_phase',sp[fam])]:rows.append({'field':m['name'],'family':fam,'sample':sample,**s})
 D=pd.DataFrame(rows);D.to_csv(out/'field_metrics.csv',index=False);summary={'status':'REAL_DR11_BISPECTRUM_PHASE','n_fields':48,'total_rows':int(P['total_rows']),'primary_family':'all','families':{}}
 for fam,g in D.groupby('family'):
  W=g.pivot(index='field',columns='sample',values=['phase_lock','bicoherence','signed_bicoherence'])
  fs={}
  for metric in ['phase_lock','bicoherence','signed_bicoherence']:
   real=W[metric]['real'];phase=W[metric]['exact_phase'];fs[metric]={'real_median':float(np.median(real)),'phase_median':float(np.median(phase)),'real_minus_phase':paired(real-phase)}
  fs['n_triangles']=int(g.n_triangles.iloc[0]);summary['families'][fam]=fs
 p=summary['families']['all']['bicoherence']['real_minus_phase'];summary['primary_decision']='PASS_HIGHER_ORDER_PHASE_COUPLING' if p['median']>0 and p['wilcoxon_p_one_sided']<.05 else 'FAIL_OR_UNCERTAIN_PHASE_COUPLING'
 (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
