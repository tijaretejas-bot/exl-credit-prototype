"""Scoring logic for the prototype. Mirrors notebooks Step 1 (baseline), Step 2 (new scorecard) and Step 4 (reasons)."""
import json
import os

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(__file__), 'data')
CFG = json.load(open(os.path.join(D, 'config.json')))
NEW = CFG['new']
BASE = CFG['baseline']
NEW_PTS = pd.read_csv(os.path.join(D, 'new_points.csv'))
BASE_PTS = pd.read_csv(os.path.join(D, 'baseline_points.csv'))
KEEP = NEW['keep']
VARS = NEW['vars']
NEUTRAL = NEW['neutral_points']
LOOKUP = {v: NEW_PTS[NEW_PTS.column == v].set_index('band').points.to_dict() for v in KEEP}
MEDIAN = {v: float(np.median(list(LOOKUP[v].values()))) for v in KEEP}
BLOOKUP = {v: BASE_PTS[BASE_PTS.column == v].set_index('band').points.to_dict() for v in BASE['vars']}
AGRI = ('Farmer (cultivator)', 'Agri labourer', 'Dairy/livestock')
R = CFG['annual_rate'] / 12

BORROWER = {
    'bureau_score': ('Your credit score is low', 'Repay existing loans on time; your score will improve'),
    'bureau_max_dpd_12m': ('Some past loan EMIs were paid late', 'Clear any overdue amounts on current loans'),
    'bureau_enquiries_6m': ('You applied for credit at many places recently', 'Avoid applying to several lenders at once'),
    'foir_declared': ('Total EMIs are high compared with your income', 'Choose a smaller loan or a longer tenure'),
    'emi_to_lean_surplus': ('The EMI is more than you can spare in your leanest months', 'A smaller loan, or EMIs matched to your earning season'),
    'emi_to_avg_surplus': ('The EMI is high compared with what you save each month', 'Choose a smaller loan amount'),
    'inflow_cv_12m': ('Your income varies a lot from month to month', 'Receive more of your income regularly into your bank/UPI account'),
    'lean_3m_inflow_ratio': ('Your income falls sharply in some months', 'Keep some savings for lean months'),
    'bounce_rate_12m': ('Some payments from your bank account bounced', 'Keep enough balance before EMIs or auto-debits'),
    'avg_eom_balance_months': ('You keep little savings in your bank account', 'Build a small balance in your account'),
    'utility_bill_ontime_rate': ('Electricity bills were often paid late', 'Pay electricity bills on time'),
    'mobile_recharge_regularity': ('Mobile recharges are irregular', 'Recharge your mobile regularly'),
    'udhaar_ontime_rate': ('Shop or dealer credit (udhaar) was often repaid late', 'Repay shop and dealer credit on time'),
    'shg_attendance_rate': ('SHG meeting attendance is low', 'Attend SHG meetings regularly'),
    'shg_savings_months': ('SHG savings are low', 'Save regularly in your SHG'),
    'drought_exposure': ('Your farm income is exposed to drought without insurance', 'Crop insurance (PMFBY) reduces this risk'),
}
STRENGTH = {
    'bureau_score': 'a good credit score', 'bureau_max_dpd_12m': 'no recent late payments on loans',
    'bureau_enquiries_6m': 'not chasing many lenders', 'foir_declared': 'low total EMIs for your income',
    'emi_to_lean_surplus': 'the EMI fits even your leanest months', 'emi_to_avg_surplus': 'the EMI fits comfortably in your monthly savings',
    'inflow_cv_12m': 'steady income', 'lean_3m_inflow_ratio': 'income holds up in lean months',
    'bounce_rate_12m': 'very few bounced payments', 'avg_eom_balance_months': 'savings kept in your bank account',
    'utility_bill_ontime_rate': 'electricity bills paid on time', 'mobile_recharge_regularity': 'regular mobile recharges',
    'udhaar_ontime_rate': 'shop/dealer credit repaid on time', 'shg_attendance_rate': 'regular SHG attendance',
    'shg_savings_months': 'good SHG savings', 'drought_exposure': 'income not exposed to drought',
}
NO_DATA = {
    'No bureau file': 'No credit history yet. Not held against you; we used your other records',
    'No digital footprint': 'Little bank/UPI activity. Receiving payments through UPI could help next time',
    'No utility data': 'No electricity bill record. Sharing your bills could help',
    'No telecom data': 'No mobile recharge record. Sharing it could help',
    'No Udhaar Passport': 'No shop/dealer credit record. An Udhaar Passport could help',
    'Not an SHG member': 'Not an SHG member. Joining an SHG could help',
}
PCT = ('bounce_rate_12m', 'utility_bill_ontime_rate', 'udhaar_ontime_rate', 'shg_attendance_rate',
       'mobile_recharge_regularity', 'foir_declared')


def _band(x, cuts, missing=None):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return missing or 'Missing'
    c = [-np.inf if v is None and i == 0 else (np.inf if v is None else v) for i, v in enumerate(cuts)]
    i = int(np.searchsorted(c, x, side='left'))
    i = min(max(i, 1), len(c) - 1)
    return str(pd.Interval(float(c[i - 1]), float(c[i]), closed='right'))


def isnan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def derive(a):
    """Add loan-dependent and engineered fields to an applicant dict."""
    a = dict(a)
    a['emi'] = round(a['loan_amount'] * R / (1 - (1 + R) ** (-a['tenure_months'])))
    a['foir_declared'] = round((a['existing_emi_bureau'] + a['emi']) / a['declared_monthly_income'], 3)
    other = a['existing_emi_bureau'] + (0 if isnan(a.get('informal_emi_declared')) else a['informal_emi_declared'])
    a['emi_to_lean_surplus'] = a['emi'] / max(a['field_lean_month_surplus'] - other, 500)
    a['emi_to_avg_surplus'] = a['emi'] / max(a['field_avg_monthly_surplus'], 500)
    agri = a['occupation'] in AGRI
    if not agri:
        a['drought_exposure'] = 'Not agri-dependent'
    elif not a['drought_flag']:
        a['drought_exposure'] = 'Agri, normal monsoon'
    elif a['crop_insured']:
        a['drought_exposure'] = 'Agri, drought but insured'
    else:
        a['drought_exposure'] = 'Agri, drought, uninsured'
    return a


def fmt(v, x):
    if isnan(x):
        return 'no data'
    if v in PCT:
        return f'{x:.0%}'
    if v in ('emi_to_lean_surplus', 'emi_to_avg_surplus'):
        return f'{x:.2f}x'
    if v in ('bureau_score', 'bureau_max_dpd_12m', 'bureau_enquiries_6m'):
        return f'{x:.0f}'
    if isinstance(x, str):
        return x
    return f'{x:.2f}'


def new_points(a):
    rows = []
    for v in KEEP:
        meta = VARS[v]
        b = str(a[v]) if meta['cuts'] is None else _band(a.get(v), meta['cuts'], meta['missing'])
        p = LOOKUP[v].get(b, MEDIAN[v])
        rows.append({'var': v, 'Signal': meta['label'], 'Value': fmt(v, a.get(v)), 'Band': b,
                     'Points': p, 'vs average': p - NEUTRAL})
    return pd.DataFrame(rows)


def knockouts(a, baseline=False):
    if baseline and not a['has_bureau_record']:
        return 'No bureau file'
    if not isnan(a.get('bureau_max_dpd_12m')) and a['bureau_max_dpd_12m'] >= 90:
        return '90+ DPD in last 12 months'
    if a['foir_declared'] > 0.50:
        return 'FOIR above 50%'
    return ''


def cut_for(a):
    return NEW['cut_thin_file'] if a['bureau_tradelines'] <= 1 else NEW['cut']


def score_new(a, top=3):
    a = derive(a)
    pts = new_points(a)
    score = int(round(pts.Points.sum()))
    ko = knockouts(a)
    cut = cut_for(a)
    approved = ko == '' and score >= cut
    pts['lost'] = NEUTRAL - pts.Points
    reasons = pts[pts.lost >= 1].sort_values('lost', ascending=False).head(top)
    strengths = pts[(pts.lost <= -1) & ~pts.Band.isin(NO_DATA)].sort_values('lost').head(top)
    rlist, slist = [], []
    for r in reasons.itertuples():
        nod = r.Band in NO_DATA
        rlist.append({'officer': (f'{r.Signal}: no data' if nod else f"{VARS[r.var]['reason']} ({r.Signal} = {r.Value})"),
                      'points': -r.lost, 'borrower': NO_DATA[r.Band] if nod else BORROWER[r.var][0],
                      'help': '' if nod else BORROWER[r.var][1]})
    for r in strengths.itertuples():
        slist.append({'officer': f'{r.Signal} = {r.Value}', 'points': -r.lost, 'borrower': STRENGTH[r.var]})
    pd_est = 1 / (1 + np.exp((score - (NEW['base'] - NEW['pdo'] / np.log(2) * np.log(NEW['odds']))) / (NEW['pdo'] / np.log(2))))
    return {'applicant': a, 'points': pts, 'score': score, 'cut': cut, 'knockout': ko, 'approved': approved,
            'reasons': rlist, 'strengths': slist, 'pd': float(pd_est)}


def score_baseline(a):
    a = derive(a)
    ko = knockouts(a, baseline=True)
    rows = []
    for v, cuts in BASE['vars'].items():
        b = _band(a.get(v), cuts)
        p = BLOOKUP[v].get(b, np.nan)
        if v == 'bureau_active_loans' and np.isnan(p):
            p = float(np.median(list(BLOOKUP[v].values())))
        rows.append({'Signal': BASE['labels'][v], 'Value': fmt(v, a.get(v)), 'Band': b, 'Points': p})
    pts = pd.DataFrame(rows)
    score = None if ko else int(round(pts.Points.sum()))
    return {'points': pts, 'score': score, 'cut': BASE['cut'], 'knockout': ko,
            'approved': ko == '' and score is not None and score >= BASE['cut']}


def counter_offer(a):
    if not isnan(a.get('bureau_max_dpd_12m')) and a['bureau_max_dpd_12m'] >= 90:
        return None
    for amt in range(int(a['loan_amount']) - 1000, 9999, -1000):
        b = dict(a); b['loan_amount'] = amt
        d = derive(b)
        if d['foir_declared'] <= 0.50 and d['emi_to_avg_surplus'] <= 1.0:
            r = score_new(b)
            if r['approved']:
                return amt, d['emi']
    return None


def borrower_message(res, offer=None):
    a = res['applicant']
    if res['knockout']:
        return f"We could not approve your loan right now. Main reason: {res['knockout']}."
    if res['approved']:
        s = ', '.join(x['borrower'] for x in res['strengths'])
        return f"Good news, your loan of Rs {a['loan_amount']:,.0f} is approved. What helped most: {s}."
    lines = [f"We could not approve a loan of Rs {a['loan_amount']:,.0f} right now. The main reasons:"]
    for i, r in enumerate(res['reasons'], 1):
        lines.append(f"{i}. {r['borrower']}." + (f" What would help: {r['help']}." if r['help'] else ''))
    if offer:
        lines.append(f"We can approve Rs {offer[0]:,.0f} today, with an EMI of Rs {offer[1]:,.0f}.")
    return '\n'.join(lines)


INPUT_FIELDS = ['occupation', 'is_rural', 'gender', 'age', 'has_bureau_record', 'bureau_tradelines', 'bureau_score',
                'bureau_max_dpd_12m', 'bureau_enquiries_6m', 'bureau_active_loans', 'existing_emi_bureau',
                'declared_monthly_income', 'loan_amount', 'tenure_months', 'field_avg_monthly_surplus',
                'field_lean_month_surplus', 'informal_emi_declared', 'inflow_cv_12m', 'lean_3m_inflow_ratio',
                'bounce_rate_12m', 'avg_eom_balance_months', 'utility_bill_ontime_rate', 'mobile_recharge_regularity',
                'udhaar_ontime_rate', 'shg_attendance_rate', 'shg_savings_months', 'drought_flag', 'crop_insured']


def from_row(row):
    a = {k: row[k] for k in INPUT_FIELDS}
    for k, v in a.items():
        if isinstance(v, (np.floating, np.integer)):
            a[k] = v.item()
    return a
