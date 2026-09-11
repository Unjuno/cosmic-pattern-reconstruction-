#!/usr/bin/env python3
"""REAL DR11 analytic covariance reconstruction after pixel-selection residualization.

Uses exactly the 36 selection-qualified bricks established in PR24.  A fixed
sky-field split is declared before fitting: 18 train / 9 validation / 9 test
(fields selected only by locked list index modulo 4).  Official g/r/i/z
selection models are cross-fitted within the train set and trained on the 18
train bricks only for validation/test residualization.

From fully valid non-overlapping 8x8 (=3.75 arcmin) selection-residual patches,
learn two Gaussian covariance models:
  1) unrestricted 64x64 empirical patch covariance;
  2) isotropic covariance obtained by averaging covariance entries at equal
     pixel separation, followed by PSD projection.
The central 4x4 (=1.875 arcmin) cells are hidden.  Diagonal regularization is
chosen on validation MSE only.  A two-sided 90% prediction-width multiplier is
also calibrated on validation normalized residuals, then evaluated blindly on
nine test bricks.

Primary PASS: isotropic covariance hidden-cell MSE is lower than the train-mean
baseline in >=7/9 test bricks AND one-sided sign and Wilcoxon p<.05.  Secondary:
compare unrestricted vs isotropic covariance and test calibrated 90% coverage.
No simulated cosmology or mock catalog is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

import selection_residual_locality_decay as sl
import cross_tracer_selection_residual_bispectrum as cs

PATCH = 8
HIDDEN = np.zeros((PATCH, PATCH), bool)
HIDDEN[2:6, 2:6] = True
HIDX = np.flatnonzero(HIDDEN.ravel())
OIDX = np.flatnonzero(~HIDDEN.ravel())
LAMBDAS = [0.01, 0.03, 0.1, 0.3, 1.0]


def residual_map(data, name, model):
    d = data[name]
    v = d['valid']
    counts = d['counts']
    mu = float(np.mean(counts[v])) + 1e-6
    pred = np.ones_like(counts, dtype=float)
    pred[v] = np.clip(model.predict(d['sel'][v]), 0.03, 20.0)
    lam = pred * mu
    z = np.zeros_like(counts, dtype=float)
    z[v] = (counts[v] - lam[v]) / np.sqrt(lam[v] + 1.0)
    z = cs.core.robust_norm(z, v, False)
    return z, v, float(np.corrcoef((counts[v]/mu), pred[v])[0,1]) if np.std(pred[v]) else float('nan')


def extract_full_valid_patches(z, valid, field):
    xs = []
    rows = []
    for iy in range(8):
        for ix in range(8):
            y0, x0 = iy*PATCH, ix*PATCH
            vv = valid[y0:y0+PATCH, x0:x0+PATCH]
            if not np.all(vv):
                continue
            p = z[y0:y0+PATCH, x0:x0+PATCH]
            xs.append(p.ravel())
            rows.append({'field':field,'patch_y':iy,'patch_x':ix})
    if not xs:
        return np.empty((0, PATCH*PATCH)), rows
    return np.asarray(xs, float), rows


def crossfit_train_residuals(data, train_names):
    maps = {}
    metrics = []
    groups = {n:i % 3 for i,n in enumerate(train_names)}
    for fold in range(3):
        test = [n for n in train_names if groups[n] == fold]
        fit = [n for n in train_names if groups[n] != fold]
        model = cs.fit_selection_model(data, fit, 'counts', 17000 + fold)
        for name in test:
            z,v,r = residual_map(data,name,model)
            maps[name]=(z,v)
            metrics.append({'field':name,'role':'train_oof','selection_count_corr':r})
    return maps,metrics


def empirical_cov(X):
    mu = X.mean(axis=0)
    C = np.cov(X, rowvar=False, ddof=1)
    return mu, C


def isotropize(C):
    coords = np.array([(y,x) for y in range(PATCH) for x in range(PATCH)], int)
    d2 = ((coords[:,None,:] - coords[None,:,:])**2).sum(axis=2)
    out = np.zeros_like(C)
    for d in np.unique(d2):
        m = d2 == d
        out[m] = float(np.mean(C[m]))
    out = (out + out.T) / 2
    # Project to a valid covariance cone after finite-sample radial averaging.
    w,V = np.linalg.eigh(out)
    floor = max(1e-8, 1e-6 * float(np.median(np.diag(C))))
    w = np.clip(w, floor, None)
    return (V * w) @ V.T


def cond_operator(mu, C, reg):
    diag_scale = max(float(np.mean(np.diag(C)[OIDX])), 1e-8)
    Coo = C[np.ix_(OIDX,OIDX)] + reg * diag_scale * np.eye(len(OIDX))
    Coh = C[np.ix_(OIDX,HIDX)]
    A = np.linalg.solve(Coo, Coh)
    C_hh = C[np.ix_(HIDX,HIDX)]
    C_ho = C[np.ix_(HIDX,OIDX)]
    cond = C_hh - C_ho @ A
    cond = (cond + cond.T) / 2
    var = np.clip(np.diag(cond), 1e-8, None)
    return A, var


def predict(X, mu, A):
    return mu[HIDX] + (X[:,OIDX] - mu[OIDX]) @ A


def metrics(y, p):
    err = y-p
    mse = float(np.mean(err**2))
    corr = float(np.corrcoef(y.ravel(),p.ravel())[0,1]) if np.std(p)>0 and np.std(y)>0 else float('nan')
    yt = y.mean(axis=1); pt = p.mean(axis=1)
    mean_corr = float(np.corrcoef(yt,pt)[0,1]) if np.std(pt)>0 and np.std(yt)>0 else float('nan')
    return {'mse':mse,'corr':corr,'hidden_mean_corr':mean_corr}


def paired_positive(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)];n=len(a);pos=int((a>0).sum())
    try: wp=float(wilcoxon(a,alternative='greater',zero_method='wilcox').pvalue)
    except Exception: wp=float('nan')
    return {'n':n,'positive':pos,'median':float(np.median(a)),'mean':float(np.mean(a)),
            'sign_p_one_sided':float(binomtest(pos,n,.5,alternative='greater').pvalue),
            'wilcoxon_p_one_sided':wp}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results/real_dr11/selection_residual_covariance_reconstruct36')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)

    data,prov = sl.acquire(); names=list(data)
    if len(names)!=36: raise RuntimeError(f'locked field count changed: {len(names)}')
    train=[n for i,n in enumerate(names) if i%4 in (0,1)]
    val=[n for i,n in enumerate(names) if i%4==2]
    test=[n for i,n in enumerate(names) if i%4==3]
    if (len(train),len(val),len(test))!=(18,9,9): raise RuntimeError('split invariant failed')

    train_maps,sel_metrics = crossfit_train_residuals(data,train)
    final_sel = cs.fit_selection_model(data,train,'counts',18000)
    maps=dict(train_maps)
    for role,subset in [('validation',val),('test',test)]:
        for name in subset:
            z,v,r=residual_map(data,name,final_sel);maps[name]=(z,v)
            sel_metrics.append({'field':name,'role':role,'selection_count_corr':r})

    X={};meta={}
    for role,subset in [('train',train),('validation',val),('test',test)]:
        arr=[];rows=[]
        for name in subset:
            z,v=maps[name];x,r=extract_full_valid_patches(z,v,name);arr.append(x);rows+=r
        X[role]=np.concatenate(arr) if arr else np.empty((0,64));meta[role]=rows
    if min(len(X['train']),len(X['validation']),len(X['test']))<30:
        raise RuntimeError({k:len(v) for k,v in X.items()})

    mu,Cfull=empirical_cov(X['train']); Ciso=isotropize(Cfull)
    covs={'full':Cfull,'isotropic':Ciso}
    tuning=[];chosen={}
    for model,C in covs.items():
        best=None
        for reg in LAMBDAS:
            A,var=cond_operator(mu,C,reg);pred=predict(X['validation'],mu,A);m=metrics(X['validation'][:,HIDX],pred)
            rec={'model':model,'reg':reg,**m};tuning.append(rec)
            if best is None or m['mse']<best[0]: best=(m['mse'],reg,A,var)
        chosen[model]={'reg':best[1],'A':best[2],'var':best[3]}

    # Calibrate central 90% absolute-error scale on validation only.
    for model in chosen:
        c=chosen[model];pv=predict(X['validation'],mu,c['A']);sd=np.sqrt(c['var'])[None,:]
        q=float(np.quantile(np.abs(X['validation'][:,HIDX]-pv)/sd,.90));c['q90']=q

    # Blind test aggregate and fieldwise metrics.
    rows=[]
    mean_pred=np.broadcast_to(mu[HIDX],(len(X['test']),len(HIDX))).copy()
    # map patch rows back to field slices
    test_meta=pd.DataFrame(meta['test'])
    for field in test:
        ii=np.flatnonzero(test_meta.field.to_numpy()==field);y=X['test'][ii][:,HIDX]
        base=metrics(y,mean_pred[ii])
        rows.append({'field':field,'model':'mean','reg':np.nan,'q90':np.nan,'coverage90':np.nan,
                     'mean_interval_width90':np.nan,**base})
        for model,c in chosen.items():
            p=predict(X['test'][ii],mu,c['A']);m=metrics(y,p);sd=np.sqrt(c['var'])[None,:]
            cov=float(np.mean(np.abs(y-p)<=c['q90']*sd));wid=float(np.mean(2*c['q90']*sd))
            rows.append({'field':field,'model':model,'reg':c['reg'],'q90':c['q90'],'coverage90':cov,
                         'mean_interval_width90':wid,**m})
    R=pd.DataFrame(rows);R.to_csv(out/'test_field_metrics.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'validation_tuning.csv',index=False)
    pd.DataFrame(sel_metrics).to_csv(out/'selection_model_metrics.csv',index=False)

    wide=R.pivot(index='field',columns='model',values='mse')
    iso_adv=wide['mean']-wide['isotropic'];full_adv=wide['mean']-wide['full'];iso_minus_full=wide['full']-wide['isotropic']
    primary=paired_positive(iso_adv.to_numpy())
    summary={
      'status':'REAL_DR11_SELECTION_RESIDUAL_COVARIANCE_RECONSTRUCTION',
      'field_split':{'train':train,'validation':val,'test':test},
      'n_patches':{k:int(len(v)) for k,v in X.items()},
      'patch_arcmin':3.75,'hidden_arcmin':1.875,
      'selection_residualization':'official g/r/i/z depth, NEXP, PSF-size, MASKBITS and BRICK_PRIMARY; train residual maps are 3-fold out-of-fold; validation/test predicted from train fields only',
      'models':{'full':{'chosen_reg':chosen['full']['reg'],'q90':chosen['full']['q90']},
                'isotropic':{'chosen_reg':chosen['isotropic']['reg'],'q90':chosen['isotropic']['q90']}},
      'test_medians':{},
      'paired_tests':{'mean_minus_isotropic_mse':primary,'mean_minus_full_mse':paired_positive(full_adv.to_numpy()),
                      'full_minus_isotropic_mse':paired_positive(iso_minus_full.to_numpy())},
      'predeclared_pass':'isotropic MSE lower than mean in >=7/9 test fields AND one-sided sign p<.05 AND Wilcoxon p<.05'
    }
    for model,g in R.groupby('model'):
        summary['test_medians'][model]={c:float(np.nanmedian(g[c])) for c in ['mse','corr','hidden_mean_corr','coverage90','mean_interval_width90'] if np.isfinite(g[c]).any()}
    summary['primary_decision']=('PASS_ISOTROPIC_COVARIANCE_RECONSTRUCTION' if primary['positive']>=7 and primary['sign_p_one_sided']<.05 and primary['wilcoxon_p_one_sided']<.05 else 'FAIL_OR_UNCERTAIN_ISOTROPIC_COVARIANCE_RECONSTRUCTION')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'provenance.json').write_text(json.dumps({'status':summary['status'],'source_regions':prov},indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
