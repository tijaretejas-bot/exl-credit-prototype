"""Export everything the Streamlit app needs as plain CSV/JSON (no pickles -> no version issues on deploy)."""
import json, os, shutil
import numpy as np, pandas as pd, joblib, nbformat

NB = '/home/claude/exl_synthetic/nb'
OUT = '/home/claude/exl_synthetic/credit_app/data'
os.makedirs(OUT, exist_ok=True)
os.chdir(NB)

# ---------------- scorecard config
M = joblib.load('new_models.joblib')
sc = M['scorecard']
F = M['PDO'] / np.log(2); O = M['BASE'] - F * np.log(M['ODDS'])
k = len(sc['keep'])
neutral = -(sc['lr'].intercept_[0] / k) * F + O / k
def clean(c):
    return None if c is None else [None if np.isinf(x) else float(x) for x in c]
cfg = {
    'new': {'cut': int(M['cut_score']), 'cut_thin_file': 550, 'neutral_points': float(neutral),
            'pdo': M['PDO'], 'base': M['BASE'], 'odds': M['ODDS'], 'keep': sc['keep'],
            'vars': {v: {'label': M['VARS'][v][0], 'cuts': clean(M['VARS'][v][1]), 'missing': M['VARS'][v][2],
                         'reason': M['VARS'][v][3]} for v in M['VARS']}},
    'baseline': {'cut': 544,
                 'vars': {'bureau_score': [None, 600, 650, 700, 750, 800, None],
                          'bureau_max_dpd_12m': [None, 0, 30, 60, None],
                          'bureau_enquiries_6m': [None, 0, 1, 2, None],
                          'bureau_active_loans': [None, 0, 1, 2, None],
                          'foir_declared': [None, .2, .3, .4, None]},
                 'labels': {'bureau_score': 'Bureau score', 'bureau_max_dpd_12m': 'Worst DPD (12m)',
                            'bureau_enquiries_6m': 'Credit enquiries (6m)', 'bureau_active_loans': 'Active loans',
                            'foir_declared': 'FOIR (EMI / declared income)'}},
    'annual_rate': 0.24,
}
json.dump(cfg, open(f'{OUT}/config.json', 'w'), indent=1)
shutil.copy('newmodel_scorecard_points.csv', f'{OUT}/new_points.csv')
bp = pd.read_csv('baseline_scorecard_points.csv')
inv = {v: k for k, v in cfg['baseline']['labels'].items()}
bp['column'] = bp.variable.map(inv)
bp.to_csv(f'{OUT}/baseline_points.csv', index=False)

# ---------------- test applicants
b = pd.read_csv('borrowers.csv')
t = (b[b.sample_split == 'test_out_of_time']
     .merge(pd.read_csv('baseline_test_scores.csv'), on='borrower_id')
     .merge(pd.read_csv('newmodel_test_scores.csv'), on='borrower_id'))
t.to_csv(f'{OUT}/applicants_2024.csv', index=False)

# ---------------- early warning: rerun step 6 code, then score every 2024 loan-month with reasons/actions
nb = nbformat.read('step6_early_warning.ipynb', 4)
parts = []
for c in nb.cells:
    if c.cell_type != 'code' or 'joblib.dump' in c.source:
        continue
    parts.append(c.source.split('latest = ')[0])          # keep REASON/action definitions, drop the snapshot
src = '\n'.join(parts)
src = src.replace('from google.colab import files; files.upload()', 'pass')
ns = {}
exec(src, ns)
allte, lr, scl, FEATS, RED, AMBER = ns['allte'], ns['lr'], ns['sc'], ns['FEATS'], ns['RED'], ns['AMBER']
REASON, action = ns['REASON'], ns['action']
Z = (allte[FEATS].astype(float).values - scl.mean_) / scl.scale_
C = pd.DataFrame(lr.coef_[0] * Z, columns=FEATS, index=allte.index).drop(columns=['month_on_book', 'has_udhaar'])
def top2(row):
    r = row[row > 0.15].sort_values(ascending=False).head(2)
    return [REASON[f] for f in r.index]
rs = C.apply(top2, axis=1)
allte['reason_1'] = rs.str[0]; allte['reason_2'] = rs.str[1]
allte['action'] = [action(f, r, o) for f, r, o in zip(allte.flag, rs, allte.occupation)]
allte.loc[allte.dpd >= 30, ['flag', 'action']] = ['In collections (30+ DPD)', 'Collections process']
keep = ['borrower_id', 'month_on_book', 'calendar_month', 'occupation', 'is_rural', 'gender', 'emi_due', 'amount_paid',
        'payment_ratio', 'dpd', 'monthly_inflow', 'avg_monthly_inflow_12m', 'season_index', 'inflow_vs_season',
        'upi_txn_count', 'new_external_loan_flag', 'bureau_enquiry_flag', 'udhaar_tab_change_pct',
        'rainfall_shock_flag', 'p', 'flag', 'reason_1', 'reason_2', 'action', 'default_90dpd_12m']
allte[keep].round(4).to_csv(f'{OUT}/ews_panel_2024.csv', index=False)
json.dump({'red': float(RED), 'amber': float(AMBER), 'features': FEATS, 'coef': lr.coef_[0].tolist(),
           'intercept': float(lr.intercept_[0]), 'mean': scl.mean_.tolist(), 'scale': scl.scale_.tolist()},
          open(f'{OUT}/ews_params.json', 'w'), indent=1)

# ---------------- results tables
for f in ['step7_results_table.csv', 'step3_signal_ladder.csv', 'step3_swap_set.csv', 'step3_score_bands.csv',
          'step5_fairness_summary.csv', 'step6_ews_flag_performance.csv', 'step6_ews_summary.csv',
          'step3_ranking_power.csv']:
    shutil.copy(f, f'{OUT}/{f}')
print('exported:', sorted(os.listdir(OUT)))
print({f: os.path.getsize(f'{OUT}/{f}') // 1024 for f in os.listdir(OUT)}, 'KB')
