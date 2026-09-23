#!/usr/bin/env python3
"""Stats on baselines comparison (standalone preprocessing methods)."""
import pandas as pd
import numpy as np
from scipy.special import betainc
from scipy.stats import t as tdist

def rm_anova(mat):
    n, k = mat.shape
    gm, sm, tm = mat.mean(), mat.mean(1), mat.mean(0)
    ss_tr = n * ((tm-gm)**2).sum()
    ss_sub = k * ((sm-gm)**2).sum()
    ss_tot = ((mat-gm)**2).sum()
    ss_err = ss_tot - ss_sub - ss_tr
    df_tr, df_err = k-1, (n-1)*(k-1)
    F = (ss_tr/df_tr) / (ss_err/df_err)
    x = df_err/(df_err + df_tr*F)
    p = float(betainc(df_err/2., df_tr/2., x))
    return F, p, df_tr, df_err

def posthoc(mat, names, a=0.05):
    n, k = mat.shape
    nc = k*(k-1)//2
    ac = a/nc
    res = []
    for i in range(k):
        for j in range(i+1, k):
            d = mat[:,i]-mat[:,j]
            se = d.std(ddof=1)/np.sqrt(n)
            t = d.mean()/se if se>1e-12 else 0.
            pu = 2.*tdist.sf(abs(t), n-1) if se>1e-12 else 1.
            pc = min(pu*nc, 1.)
            res.append((f"{names[i]} vs {names[j]}", d.mean(), t, pu, pc, pc<ac))
    return res, ac

pp = ["none","SNV","SG1","EMSC","Detrend"]
cl = {"svm":"LinearSVM","rbf_svm":"RBF-SVM","rf":"RF"}
vl = {"A_postSNV":"A (post-SNV)","A2_noPostSNV":"A2 (trad PCA)"}
df = pd.read_csv(r"d:\PCA\results_baselines_raw.csv")

# ─── Per-(variant,classifier) stats ───
for v in ["A_postSNV","A2_noPostSNV"]:
    for ct in ["svm","rbf_svm","rf"]:
        sub = df[(df.variant==v) & (df.classifier==ct)]
        piv = sub.pivot_table(index='repeat',columns='preprocess',values='OA',aggfunc='first')[pp]
        mat = piv.values
        F, p, dft, dfe = rm_anova(mat)
        S = " *** SIG" if p<0.05 else ""
        print(f"\n{'='*80}")
        print(f"[{vl[v]} + {cl[ct]}]  RM-ANOVA  F({dft},{dfe})={F:.4f}  p={p:.6f}{S}")
        ph, ac = posthoc(mat, pp)
        print(f"\n  Post-hoc (Bonferroni, α_corr={ac:.4f}):")
        print(f"  {'Pair':<30} {'Δ(OA)%':>8} {'t':>8} {'p_unc':>10} {'p_corr':>10} {'Sig':>6}")
        print(f"  {'-'*70}")
        for pair, md, t, pu, pc, s in ph:
            print(f"  {pair:<30} {md*100:>7.2f}% {t:>8.3f} {pu:>10.6f} {pc:>10.6f}{'  ***' if s else ''}")

# ─── Summary ───
print(f"\n{'='*80}")
print("SUMMARY: RM-ANOVA F-values")
print(f"{'='*80}")
print(f"{'Variant':<18} {'Classifier':<14} {'F(4,96)':>12} {'p':>12} {'Sig':>6}")
print(f"{'-'*60}")
for v in ["A_postSNV","A2_noPostSNV"]:
    for ct in ["svm","rbf_svm","rf"]:
        sub = df[(df.variant==v) & (df.classifier==ct)]
        piv = sub.pivot_table(index='repeat',columns='preprocess',values='OA',aggfunc='first')[pp]
        F, p, _, _ = rm_anova(piv.values)
        print(f"{vl[v]:<18} {cl[ct]:<14} {F:>12.4f} {p:>12.6f}{'  ***' if p<0.05 else ''}")

# ─── Key comparison table: A vs best traditional ───
print(f"\n{'='*80}")
print("KEY: USP (none+postSNV) vs best traditional baseline per classifier")
print(f"{'='*80}")
print(f"{'Classifier':<14} {'USP OA':>10} {'Best Trad':>12} {'OA':>10} {'Δ':>8} {'p_corr':>10}")
print(f"{'-'*65}")
for ct in ["svm","rbf_svm","rf"]:
    sub_a = df[(df.variant=="A_postSNV") & (df.classifier==ct) & (df.preprocess=="none")]
    sub_a2 = df[(df.variant=="A2_noPostSNV") & (df.classifier==ct)]
    oa_usp = sub_a.OA.mean()
    # best traditional
    tm = sub_a2.groupby('preprocess').OA.mean()
    best_trad = tm.idxmax()
    oa_best = tm.max()
    # paired t-test
    piv_a2 = sub_a2.pivot_table(index='repeat',columns='preprocess',values='OA',aggfunc='first')
    d = sub_a.OA.values - piv_a2[best_trad].values
    se = d.std(ddof=1)/np.sqrt(len(d))
    t = d.mean()/se if se>1e-12 else 0.
    pu = 2.*tdist.sf(abs(t), len(d)-1) if se>1e-12 else 1.
    print(f"{cl[ct]:<14} {oa_usp*100:>9.2f}% {best_trad:>12} {oa_best*100:>9.2f}% {d.mean()*100:>7.2f}% {pu:>10.6f}")
