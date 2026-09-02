#!/usr/bin/env python3
"""REAL DR11 PSF-like vs extended cross-bispectrum phase-coupling test.

Uses the same objective availability protocol as the accepted 36-field
morphology tracer-split stress. Per brick, PSF and extended (REX/EXP/DEV/SER)
subsets are equalized in source count with a fixed RNG. Science inputs after
subsetting are positions only.

Primary statistic symmetrizes the two mixed closure products
  P(k1) P(k2) E*(k1+k2) and E(k1) E(k2) P*(k1+k2).
The paired null preserves each tracer's exact Fourier amplitude but randomizes
the phase of the opposite tracer used at k3. No simulated cosmology is used.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

import tracer_split_validate as tr
import bispectrum_phase_validate as bp

TARGET_FIELDS=36; MAX_CANDIDATES=48; MIN_PER_TRACER=1000; GRID=64

def mixed_stats(a,b,tri):
    A=np.fft.fft2(a); B=np.fft.fft2(b); out={}
    for name,T in tri.items():
        x1,y1,x2,y2,x3,y3=[T[:,j] for j in range(6)]
        a1=A[np.mod(y1,GRID),np.mod(x1,GRID)]; a2=A[np.mod(y2,GRID),np.mod(x2,GRID)]; a3=A[np.mod(y3,GRID),np.mod(x3,GRID)]
        b1=B[np.mod(y1,GRID),np.mod(x1,GRID)]; b2=B[np.mod(y2,GRID),np.mod(x2,GRID)]; b3=B[np.mod(y3,GRID),np.mod(x3,GRID)]
        c1=a1*a2*np.conj(b3); c2=b1*b2*np.conj(a3)
        def one(c,u1,u2,v3):
            unit=c/np.where(abs(c)>0,abs(c),1)
            den=np.sqrt(np.mean(abs(u1*u2)**2)*np.mean(abs(v3)**2))+1e-30
            return abs(np.mean(unit)),abs(np.mean(c))/den,np.real(np.mean(c))/den
        p1,z1,s1=one(c1,a1,a2,b3); p2,z2,s2=one(c2,b1,b2,a3)
        out[name]={'phase_lock':float((p1+p2)/2),'bicoherence':float((z1+z2)/2),'signed_bicoherence':float((s1+s2)/2),'n_triangles':int(len(T))}
    return out

def build_data(centers):
    regs=centers.get('regions',[])[:MAX_CANDIDATES]; data={}; prov=[]; rejected=[]; used=set(); rng=np.random.default_rng(20260831)
    for j,r in enumerate(regs):
        if len(data)>=TARGET_FIELDS: break
        name=r['name']; ra=float(r['center_ra_deg']); dec=float(r['center_dec_deg'])
        try: brick,bq=tr.choose_brick(ra,dec)
        except Exception as e: rejected.append({'field':name,'reason':'brick_resolution','error':str(e)}); continue
        if brick in used: rejected.append({'field':name,'brick':brick,'reason':'duplicate'}); continue
        used.add(brick); print(f'[cross-bispec] candidate {j+1}/{MAX_CANDIDATES} {name}->{brick}',flush=True)
        try: d,p=tr.get_tractor(brick)
        except Exception as e: rejected.append({'field':name,'brick':brick,'reason':'tractor_acquisition','error':str(e)}); continue
        psf=d[d.type.eq('PSF')].copy(); ext=d[d.type.isin(tr.EXT_TYPES)].copy(); n=min(len(psf),len(ext))
        if n<MIN_PER_TRACER:
            rejected.append({'field':name,'brick':brick,'reason':'tracer_availability','n_psf':int(len(psf)),'n_extended':int(len(ext))}); continue
        psf_eq=psf.iloc[rng.choice(len(psf),n,replace=False)]; ext_eq=ext.iloc[rng.choice(len(ext),n,replace=False)]
        gp=bp.gaussianize(tr.grid_from(psf_eq)); ge=bp.gaussianize(tr.grid_from(ext_eq))
        data[name]={'psf':gp,'ext':ge}; prov.append({'field':name,'brick':brick,'brick_choice_query':bq,'tractor':p,'n_psf':int(len(psf)),'n_extended':int(len(ext)),'equalized_n':int(n)})
        print(f'[cross-bispec] accept {len(data)}/{TARGET_FIELDS}: eq={n}',flush=True)
    return data,prov,rejected

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--centers',default='data/real/dr11/expanded48/provenance.json'); ap.add_argument('--out',default='results/real_dr11/cross_tracer_bispectrum36'); args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    C=json.loads(Path(args.centers).read_text())
    if C.get('status')!='REAL_DR11': raise RuntimeError('REAL_DR11 centers required')
    data,prov,rejected=build_data(C)
    if len(data)!=TARGET_FIELDS:
        (out/'availability_rejections.json').write_text(json.dumps(rejected,indent=2,sort_keys=True)+'\n'); raise RuntimeError(f'only {len(data)} accepted')
    tri=bp.build_triangles(seed=20260901,max_pairs=16000); rows=[]
    for i,(name,d) in enumerate(data.items()):
        psf,ext=d['psf'],d['ext']; psf_phase=bp.exact_phase(psf,20264001+i); ext_phase=bp.exact_phase(ext,20265001+i)
        real=mixed_stats(psf,ext,tri)
        # Symmetric null: average stats from scrambling EXT and scrambling PSF independently.
        n1=mixed_stats(psf,ext_phase,tri); n2=mixed_stats(psf_phase,ext,tri)
        for fam in tri:
            rows.append({'field':name,'family':fam,'sample':'real',**real[fam]})
            rows.append({'field':name,'family':fam,'sample':'phase_null','phase_lock':float((n1[fam]['phase_lock']+n2[fam]['phase_lock'])/2),'bicoherence':float((n1[fam]['bicoherence']+n2[fam]['bicoherence'])/2),'signed_bicoherence':float((n1[fam]['signed_bicoherence']+n2[fam]['signed_bicoherence'])/2),'n_triangles':real[fam]['n_triangles']})
    D=pd.DataFrame(rows); D.to_csv(out/'field_metrics.csv',index=False)
    summary={'status':'REAL_DR11_CROSS_TRACER_BISPECTRUM','n_fields':36,'subsets':'count-equalized PSF vs REX/EXP/DEV/SER; positions only after morphology split','primary_family':'all','availability_rejections':rejected,'families':{}}
    for fam,g in D.groupby('family'):
        W=g.pivot(index='field',columns='sample',values=['phase_lock','bicoherence','signed_bicoherence']); fs={}
        for metric in ['phase_lock','bicoherence','signed_bicoherence']:
            real=W[metric]['real']; null=W[metric]['phase_null']; fs[metric]={'real_median':float(np.nanmedian(real)),'phase_null_median':float(np.nanmedian(null)),'real_minus_null':bp.paired(real-null)}
        fs['n_triangles']=int(g.n_triangles.iloc[0]); summary['families'][fam]=fs
    p=summary['families']['all']['bicoherence']['real_minus_null']
    summary['primary_decision']='PASS_SHARED_CROSS_TRACER_PHASE_COUPLING' if p['median']>0 and p['wilcoxon_p_one_sided']<.05 and p['sign_p_one_sided']<.05 else 'FAIL_OR_UNCERTAIN_SHARED_CROSS_TRACER_PHASE_COUPLING'
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); (out/'provenance.json').write_text(json.dumps({'status':summary['status'],'regions':prov,'rejections':rejected},indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
