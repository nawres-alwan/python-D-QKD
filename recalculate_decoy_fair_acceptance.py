#!/usr/bin/env python3
import csv, json, math, os, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unified_decoy import kappa_family, sector_observed
from sector_geat import key_length, h2, ALPHA_GRID

OUT=Path(os.environ.get('DQKD_OUT','.')); OUT.mkdir(parents=True,exist_ok=True)
N=1e12; M=8; PM=np.full(M,1/M); AF=0.2; ETADET=0.6; PD=1e-7; FEC=1.16
INT=(0.9,0.02,0.001); PMU=np.array([0.8,0.15,0.05]); PIN=0.5
EPSSEC=1e-8; EPSCOM=1e-10; K=8*3*2*3; TAU=2e-5
GAM=np.logspace(-3,np.log10(0.4),14)
E0=np.array([.004,.006,.008,.010,.014,.018,.024,.030])
KAP=np.array([.00005,.0001,.0002,.0005,.002,.004,.007,.010])

def prof(L): return np.clip(E0+KAP*L,1e-4,.499)
def widths(g,q):
 n=max(int(g*N),1); t=math.log(2*K/EPSCOM); q=np.clip(q,0,1)
 return np.sqrt(2*q*(1-q)*t/n)+2*t/(3*n)
def cells(eta,e):
 q=np.zeros((M,3,6))
 for m in range(M):
  for j,mu in enumerate(INT):
   o=sector_observed(mu,eta,PD,float(e[m])); q[m,j]=PM[m]*PMU[j]*PIN*np.r_[o,o]
 return q

def fam(eta,e): return kappa_family(eta,PD,float(e),fw_iters=30)

def rate(eta,e,kept,sector_fams=None,pool_fam=None,pooled=False):
 kept=tuple(kept); pk=PM[list(kept)].sum(); q=cells(eta,e); best=None
 fs=[pool_fam] if pooled else [sector_fams[m] for m in kept]; nk=min(map(len,fs))
 for ki in range(nk):
  ds=[pool_fam[ki]] if pooled else [sector_fams[m][ki] for m in kept]; p1=ds[0]['p1']
  for g in GAM:
   pref=(1-g)**2*p1; comp=np.zeros_like(q)
   if pooled:
    d=ds[0]; honest=pk*pref*(d['value_hon']-TAU)
    for m in kept:
     for j,mu in enumerate(INT): comp[m,j]=pref*np.asarray(d['lam'][mu])/(PMU[j]*PIN)
    om=sum((PM[m]/pk)*sector_observed(INT[0],eta,PD,float(e[m])) for m in kept)
    Q=om[0]+om[1]; E=om[0]/max(Q,1e-15); lec=FEC*N*(1-g)**2*pk*Q*h2(np.clip(E,1e-12,.5))
   else:
    honest=0.; xec=0.
    for d,m in zip(ds,kept):
     honest+=PM[m]*pref*(d['value_hon']-TAU)
     for j,mu in enumerate(INT): comp[m,j]=pref*np.asarray(d['lam'][mu])/(PMU[j]*PIN)
     o=sector_observed(INT[0],eta,PD,float(e[m])); Q=o[0]+o[1]; E=o[0]/max(Q,1e-15); xec+=PM[m]*Q*h2(np.clip(E,1e-12,.5))
    lec=FEC*N*(1-g)**2*xec
   pen=float(np.sum(np.abs(comp)*widths(g,q))); acc=honest-pen
   if acc<=0: continue
   rng=float(comp.max()-comp.min()); ls=key_length(N,acc,rng,g,ALPHA_GRID,lec,eps_secure=EPSSEC,dA=2)
   i=int(np.argmax(ls)); r=max(float(ls[i]),0)/N
   cand=dict(rate=r,gamma=float(g),alpha=float(ALPHA_GRID[i]),penalty=pen,honest=honest,accepted=acc,range=rng)
   if best is None or r>best['rate']: best=cand
 return best or dict(rate=0.,gamma=float('nan'),alpha=float('nan'),penalty=float('nan'),honest=float('nan'),accepted=float('nan'),range=float('nan'))

def solve(L):
 t0=time.time(); eta=ETADET*10**(-AF*L/10); e=prof(L); order=np.argsort(e); sf={}
 for m in range(M):
  print('SDP',L,'sector',m,'e',e[m],flush=True); sf[m]=fam(eta,e[m])
 allk=tuple(range(M)); sa=rate(eta,e,allk,sector_fams=sf)
 pref=[]
 for k in range(1,M+1):
  kk=tuple(int(x) for x in order[:k]); rr=rate(eta,e,kk,sector_fams=sf); pref.append((rr,kk))
 sk,kept=max(pref,key=lambda x:x[0]['rate']); pk=PM[list(kept)].sum()
 ea=float(PM@e); ek=float((PM[list(kept)]/pk)@e[list(kept)])
 print('SDP',L,'pool all',ea,flush=True); pfa=fam(eta,ea); pa=rate(eta,e,allk,pool_fam=pfa,pooled=True)
 print('SDP',L,'pool kept',kept,ek,flush=True); pfk=fam(eta,ek); pkrt=rate(eta,e,kept,pool_fam=pfk,pooled=True)
 d=dict(distance_km=L,kept_indices=','.join(map(str,kept)),n_kept=len(kept),e_pool_all=ea,e_pool_kept=ek,
        R_pool_all=pa['rate'],R_sector_all=sa['rate'],R_pool_kept=pkrt['rate'],R_sector_kept=sk['rate'],
        selection_gain=pkrt['rate']-pa['rate'],conditioning_gain=sk['rate']-pkrt['rate'],combined_gain=sk['rate']-pa['rate'],
        gamma_pool_all=pa['gamma'],gamma_sector_all=sa['gamma'],gamma_pool_kept=pkrt['gamma'],gamma_sector_kept=sk['gamma'],runtime_s=time.time()-t0)
 print('RESULT',json.dumps(d),flush=True); return d

def output(rows):
 rows=sorted(rows,key=lambda r:r['distance_km']); p=OUT/'decoy_fair_finite_width_results.csv'
 with p.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 t=math.log(2*K/EPSCOM); ec=2*K*math.exp(-t); mr=0
 for r in rows:
  if r['distance_km']<=0: continue
  pl=-math.log2(1-10**(-AF*r['distance_km']/10))
  mr=max(mr,max(r[k] for k in ('R_pool_all','R_sector_all','R_pool_kept','R_sector_kept'))/pl)
 if mr>1+1e-9: raise RuntimeError('PLOB '+str(mr))
 txt=[f'eps_com <= {ec:.12e}',f'max rate/PLOB = {mr:.6f}',f'parameters: eta_det={ETADET}, p_dark={PD}, intensities={INT}, p_mu={tuple(PMU)}','gradient scaling: pref*lambda/(p_mu*p_input); no extra 1/p_m']
 txt += [json.dumps(r,sort_keys=True) for r in rows]; (OUT/'decoy_fair_finite_width_report.txt').write_text('\n'.join(txt)+'\n')
 x=np.array([r['distance_km'] for r in rows]); fig,ax=plt.subplots(figsize=(7,4.6))
 for k,l in [('R_pool_all','pool all'),('R_sector_all','sector all'),('R_pool_kept','pool kept'),('R_sector_kept','sector kept')]:
  y=np.array([r[k] for r in rows]); ax.semilogy(x,np.where(y>0,y,np.nan),'-o',label=l)
 ax.set(xlabel='Distance (km)',ylabel='Key rate (bits per emitted pulse)',title='Full decoy SDP with finite-width acceptance'); ax.grid(True,which='both',alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'fig_decoy_fair_sdp.pdf'); fig.savefig(OUT/'fig_decoy_fair_sdp.png',dpi=250); plt.close(fig)

def main():
 ds=[float(x) for x in os.environ.get('DQKD_DISTANCES','0,10,20,30,40,50,60,80,100,120').split(',')]
 rows=[]
 for L in ds: rows.append(solve(L)); output(rows)
if __name__=='__main__': main()
