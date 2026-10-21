"""
Inclusive Credit Decisioning - prototype
EXL Live Project, IIM Lucknow: Credit Assessment for Underserved Borrowers
Run locally:  streamlit run app.py
"""
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import scoring as S

st.set_page_config(page_title='Inclusive Credit Decisioning', page_icon='📊', layout='wide')
_HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(_HERE, 'data') if os.path.exists(os.path.join(_HERE, 'data', 'config.json')) else _HERE  # works whether data files sit in data/ or next to app.py

INK, ACCENT, OLD, GOOD, AMBER_C, RED_C = '#1F2A44', '#1F6FB2', '#9AA5B1', '#2E8B57', '#E8A33D', '#C8463D'
st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1250px;}}
h1, h2, h3 {{color: {INK};}}
.card {{border: 1px solid #E3E7EC; border-radius: 10px; padding: 14px 16px; background: #FBFCFD; height: 100%;}}
.card h4 {{margin: 0 0 6px 0; font-size: 0.95rem; color: #5A6573; font-weight: 600;}}
.big {{font-size: 1.9rem; font-weight: 700; line-height: 1.1;}}
.pill {{display:inline-block; padding: 2px 10px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; color: white;}}
.approve {{background: {GOOD};}} .decline {{background: {RED_C};}} .amber {{background: {AMBER_C};}} .grey {{background: {OLD};}}
.letter {{border-left: 4px solid {ACCENT}; background: #F4F8FC; padding: 12px 16px; border-radius: 4px; white-space: pre-line;}}
.small {{color: #5A6573; font-size: 0.85rem;}}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    apps = pd.read_csv(os.path.join(D, 'applicants_2024.csv'))
    ews = pd.read_csv(os.path.join(D, 'ews_panel_2024.csv'))
    res = pd.read_csv(os.path.join(D, 'step7_results_table.csv'))
    rank = pd.read_csv(os.path.join(D, 'step3_ranking_power.csv'))
    ladder = pd.read_csv(os.path.join(D, 'step3_signal_ladder.csv'))
    flags = pd.read_csv(os.path.join(D, 'step6_ews_flag_performance.csv'))
    return apps, ews, res, rank, ladder, flags


apps, ews, results, rank, ladder, flagperf = load()

with st.sidebar:
    st.markdown('### Inclusive Credit Decisioning')
    st.caption('Prototype for the EXL Live Project · IIM Lucknow')
    st.markdown('**Problem statement**  \nTraditional credit models exclude thin-file, rural and underbanked borrowers '
                'and miss early signs of stress. How might we enable fairer, more predictive credit decisions using '
                'broader financial and behavioural signals?')
    st.markdown('---')
    st.markdown('**Team**  \nS R Ashwin Narayan · Tejas Tijare · Utkarsh Kumar')
    st.markdown('---')
    st.caption('All data is synthetic (10,000 borrowers, 2023–24). Results show the approach works; they are not '
               'evidence from a real portfolio.')

st.title('Inclusive Credit Decisioning')
st.markdown('<span class="small">A lender-facing tool for NBFCs and MFIs: score new applicants using broader signals, '
            'explain every decision, and flag stressed loans before EMIs are missed.</span>', unsafe_allow_html=True)

tab_res, tab_app, tab_ews, tab_how = st.tabs(['📈 Results', '🧾 Score an applicant', '🚦 Early-warning portfolio', 'ℹ️ How it works'])

# =====================================================================================================  RESULTS
with tab_res:
    st.subheader('Traditional bureau scorecard vs our model')
    st.caption('2024 out-of-time test set · 3,319 applicants the models never saw while being built')

    def kpi(col, title, old, new, note):
        col.markdown(f'<div class="card"><h4>{title}</h4><span class="small">Traditional</span> '
                     f'<b style="color:{OLD}">{old}</b><br><span class="big" style="color:{ACCENT}">{new}</span>'
                     f'<br><span class="small">{note}</span></div>', unsafe_allow_html=True)

    c = st.columns(5)
    kpi(c[0], 'Predictive power (Gini)', '0.28', '0.50', '0 = random, 1 = perfect')
    kpi(c[1], 'Approval rate', '61.2%', '79.6%', 'at recommended cut-offs')
    kpi(c[2], 'Default among approved', '5.8%', '5.5%', 'more approvals, no extra risk')
    kpi(c[3], 'Thin-file approval', '46.9%', '80.8%', '80% rule: FAIL → pass')
    kpi(c[4], 'Early-warning AUC', '0.59', '0.74', '64% of defaulters flagged 1–3 months ahead')

    st.markdown('')
    left, right = st.columns(2)
    with left:
        st.markdown('**Predictive power by segment (Gini)**')
        g = rank[rank.Group.isin(['All applicants', 'Thin-file', 'Rural', 'Underbanked', 'No bureau file', 'Has bureau file'])]
        g = g.melt(id_vars='Group', value_vars=['Gini · Baseline (bureau)', 'Gini · New scorecard'],
                   var_name='Model', value_name='Gini').fillna(0)
        g['Model'] = g.Model.map({'Gini · Baseline (bureau)': 'Traditional', 'Gini · New scorecard': 'Our model'})
        ch = alt.Chart(g).mark_bar().encode(
            y=alt.Y('Group:N', sort=['All applicants', 'Thin-file', 'Rural', 'Underbanked', 'No bureau file', 'Has bureau file'], title=None,
                    axis=alt.Axis(labelLimit=220)),
            yOffset=alt.YOffset('Model:N', sort=['Traditional', 'Our model']),
            x=alt.X('Gini:Q', scale=alt.Scale(domain=[0, 0.6])),
            color=alt.Color('Model:N', scale=alt.Scale(domain=['Traditional', 'Our model'], range=[OLD, ACCENT]),
                            legend=alt.Legend(orient='bottom', title=None)),
            tooltip=['Group', 'Model', alt.Tooltip('Gini:Q', format='.2f')]).properties(height=300)
        st.altair_chart(ch, use_container_width=True)
        st.caption('Traditional cannot rank applicants with no bureau file at all; it declines every one of them.')
    with right:
        st.markdown('**What each broader signal adds (Gini, all applicants)**')
        l = ladder.copy()
        l['Signals'] = l['Signals used'].str.replace('+ ', '', regex=False)
        ch = alt.Chart(l).mark_bar(color=ACCENT).encode(
            y=alt.Y('Signals:N', sort=list(l.Signals), title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X('Gini (all):Q', scale=alt.Scale(domain=[0, 0.6]), title='Gini (cumulative)'),
            tooltip=['Signals used', alt.Tooltip('Gini (all):Q', format='.3f'), alt.Tooltip('Gain (all):Q', format='+.3f')])
        txt = ch.mark_text(align='left', dx=4, color=INK).encode(text=alt.Text('Gini (all):Q', format='.2f'))
        st.altair_chart((ch + txt).properties(height=300), use_container_width=True)
        st.caption('Each bar adds one data source to the previous ones.')

    st.markdown('**Fairness: approval-rate ratio (80% rule)**')
    fr = pd.DataFrame({'Comparison': ['Women ÷ men', 'Rural ÷ urban', 'Thin-file ÷ thick-file', 'Underbanked ÷ banked'] * 2,
                       'Model': ['Traditional'] * 4 + ['Our model'] * 4,
                       'Ratio': [0.91, 0.89, 0.62, 0.74, 1.05, 0.95, 1.03, 1.02]})
    base = alt.Chart(fr).mark_bar().encode(
        y=alt.Y('Comparison:N', title=None, sort=None, axis=alt.Axis(labelLimit=220)), yOffset=alt.YOffset('Model:N', sort=['Traditional', 'Our model']),
        x=alt.X('Ratio:Q', scale=alt.Scale(domain=[0, 1.2]), axis=alt.Axis(tickCount=6, format='.1f')),
        color=alt.Color('Model:N', scale=alt.Scale(domain=['Traditional', 'Our model'], range=[OLD, ACCENT]),
                        legend=alt.Legend(orient='bottom', title=None)),
        tooltip=['Comparison', 'Model', alt.Tooltip('Ratio:Q', format='.2f')])
    rule = alt.Chart(pd.DataFrame({'x': [0.8]})).mark_rule(color=RED_C, strokeDash=[4, 3]).encode(x='x:Q')
    st.altair_chart((base + rule).properties(height=230), use_container_width=True)
    st.caption('Dashed line = 0.80. Below it, a group is approved at under 80% of the best-treated group\'s rate.')

    with st.expander('Full results table', expanded=False):
        sec = st.selectbox('Section', ['All'] + list(results['Problem statement'].unique()))
        show = results if sec == 'All' else results[results['Problem statement'] == sec]
        st.dataframe(show, hide_index=True, use_container_width=True)

# =====================================================================================================  APPLICANT
OCCS = ['Farmer (cultivator)', 'Agri labourer', 'Dairy/livestock', 'Kirana/small trader', 'Home enterprise',
        'Gig/delivery worker', 'Salaried (informal)', 'Self-employed services']

with tab_app:
    st.subheader('Score an applicant')
    mode = st.radio('Applicant', ['Pick a 2024 applicant', 'Enter details yourself'], horizontal=True, label_visibility='collapsed')

    if mode == 'Pick a 2024 applicant':
        f1, f2, f3 = st.columns(3)
        seg = f1.selectbox('Segment', ['All', 'No bureau file', 'Thin-file', 'Rural', 'Underbanked', 'Women · rural'])
        outcome = f2.selectbox('Show', ['All', 'Approved by us, declined by bureau', 'Declined by both', 'Approved by both'])
        pool = apps.copy()
        if seg == 'No bureau file': pool = pool[pool.has_bureau_record == 0]
        elif seg == 'Thin-file': pool = pool[pool.is_thin_file == 1]
        elif seg == 'Rural': pool = pool[pool.is_rural == 1]
        elif seg == 'Underbanked': pool = pool[pool.is_underbanked == 1]
        elif seg == 'Women · rural': pool = pool[(pool.gender == 'F') & (pool.is_rural == 1)]
        rec = (pool.sc_score >= np.where(pool.bureau_tradelines <= 1, 550, 534)) & (pool.new_knockout.fillna('') == '')
        if outcome == 'Approved by us, declined by bureau': pool = pool[rec & (pool.base_approved == 0)]
        elif outcome == 'Declined by both': pool = pool[~rec & (pool.base_approved == 0)]
        elif outcome == 'Approved by both': pool = pool[rec & (pool.base_approved == 1)]
        ids = pool.borrower_id.tolist()
        if not ids:
            st.info('No applicants match these filters.'); st.stop()
        want = st.query_params.get('id', 'B09251')          # e.g. ?id=B00252 opens that applicant
        default_id = want if want in ids else ids[0]
        bid = f3.selectbox(f'Applicant ({len(ids)} match)', ids, index=ids.index(default_id))
        row = apps.set_index('borrower_id').loc[bid]
        a = S.from_row(row)
        st.caption(f"{row.occupation} · {'rural' if row.is_rural else 'urban'} · {'woman' if row.gender == 'F' else 'man'}, "
                   f"{row.age} · {row.state} · asks for Rs {row.loan_amount:,.0f} over {row.tenure_months} months · "
                   f"actually {'DEFAULTED' if row.default_90dpd_12m else 'repaid'} (hidden from the models)")
        new_amt = st.slider('Try a different loan amount (Rs)', 10000, 200000, int(row.loan_amount), 1000)
        a['loan_amount'] = new_amt
    else:
        with st.form('manual'):
            st.markdown('**Profile and loan**')
            c = st.columns(4)
            occ = c[0].selectbox('Occupation', OCCS)
            rural = c[1].selectbox('Area', ['Rural', 'Urban']) == 'Rural'
            gender = c[2].selectbox('Gender (not used by the model)', ['F', 'M'])
            age = c[3].number_input('Age', 21, 65, 35)
            c = st.columns(4)
            loan = c[0].number_input('Loan amount (Rs)', 10000, 200000, 30000, 1000)
            tenure = c[1].selectbox('Tenure (months)', [12, 18, 24], index=0)
            dinc = c[2].number_input('Declared monthly income (Rs)', 3000, 200000, 15000, 500)
            drought = c[3].selectbox('Last monsoon', ['Normal', 'Drought (< -20%)']) != 'Normal'
            st.markdown('**Credit bureau**')
            c = st.columns(5)
            hasb = c[0].selectbox('Bureau file?', ['No', 'Yes']) == 'Yes'
            tl = c[1].number_input('Tradelines', 0, 20, 2 if hasb else 0, disabled=not hasb)
            bsc = c[2].number_input('Bureau score', 300, 900, 700, disabled=not hasb)
            dpd = c[3].selectbox('Worst DPD (12m)', [0, 30, 60, 90], disabled=not hasb)
            enq = c[4].number_input('Enquiries (6m)', 0, 20, 0, disabled=not hasb)
            c = st.columns(5)
            act = c[0].number_input('Active loans', 0, 10, 0, disabled=not hasb)
            eemi = c[1].number_input('Existing EMI (Rs)', 0, 100000, 0, 100)
            fav = c[2].number_input('Field: avg monthly surplus (Rs)', -20000, 100000, 4500, 100)
            fle = c[3].number_input('Field: lean-month surplus (Rs)', -50000, 100000, 1500, 100)
            insured = c[4].selectbox('Crop insured?', ['No', 'Yes']) == 'Yes'
            st.markdown('**Broader signals** (leave a source switched off if the applicant has no such record)')
            c = st.columns(4)
            dig = c[0].toggle('Bank/UPI data', True)
            cv = c[0].slider('Income volatility', 0.0, 0.8, 0.18, 0.01, disabled=not dig)
            lean = c[0].slider('Lean-season income ratio', 0.2, 1.0, 0.8, 0.01, disabled=not dig)
            bounce = c[1].slider('Bounce rate', 0.0, 0.3, 0.04, 0.005, disabled=not dig)
            eom = c[1].slider('Savings buffer (months)', 0.0, 5.0, 0.5, 0.05, disabled=not dig)
            util_on = c[2].toggle('Electricity bills', True)
            util = c[2].slider('Bills paid on time', 0.0, 1.0, 0.88, 0.01, disabled=not util_on)
            rec_on = c[2].toggle('Mobile recharges', True)
            rech = c[2].slider('Recharge regularity', 0.0, 1.0, 0.8, 0.01, disabled=not rec_on)
            ud_on = c[3].toggle('Udhaar Passport', False)
            ud = c[3].slider('Udhaar on-time rate', 0.0, 1.0, 0.85, 0.01, disabled=not ud_on)
            iemi = c[3].number_input('Informal EMI (Rs)', 0, 50000, 0, 100, disabled=not ud_on)
            shg_on = c[3].toggle('SHG member', False)
            shga = c[3].slider('SHG attendance', 0.0, 1.0, 0.85, 0.01, disabled=not shg_on)
            shgs = c[3].slider('SHG savings (months)', 0.0, 5.0, 0.6, 0.05, disabled=not shg_on)
            st.form_submit_button('Score this applicant', type='primary')
        nan = np.nan
        a = {'occupation': occ, 'is_rural': int(rural), 'gender': gender, 'age': age,
             'has_bureau_record': int(hasb), 'bureau_tradelines': tl if hasb else 0,
             'bureau_score': bsc if hasb else nan, 'bureau_max_dpd_12m': dpd if hasb else nan,
             'bureau_enquiries_6m': enq if hasb else nan, 'bureau_active_loans': act if hasb else 0,
             'existing_emi_bureau': eemi, 'declared_monthly_income': dinc, 'loan_amount': loan, 'tenure_months': tenure,
             'field_avg_monthly_surplus': fav, 'field_lean_month_surplus': fle,
             'informal_emi_declared': iemi if ud_on else nan,
             'inflow_cv_12m': cv if dig else nan, 'lean_3m_inflow_ratio': lean if dig else nan,
             'bounce_rate_12m': bounce if dig else nan, 'avg_eom_balance_months': eom if dig else nan,
             'utility_bill_ontime_rate': util if util_on else nan, 'mobile_recharge_regularity': rech if rec_on else nan,
             'udhaar_ontime_rate': ud if ud_on else nan, 'shg_attendance_rate': shga if shg_on else nan,
             'shg_savings_months': shgs if shg_on else nan, 'drought_flag': int(drought), 'crop_insured': int(insured)}

    new = S.score_new(a)
    old = S.score_baseline(a)
    offer = S.counter_offer(a) if (not new['approved'] and new['knockout'] != '90+ DPD in last 12 months') else None
    d = new['applicant']

    st.markdown('')
    c1, c2, c3 = st.columns([1, 1.25, 1.35])
    with c1:
        pill = ('<span class="pill approve">APPROVE</span>' if old['approved'] else '<span class="pill decline">DECLINE</span>')
        sc_txt = f"{old['score']} / cut-off {old['cut']}" if old['score'] is not None else 'not scored'
        why = f"Knock-out: {old['knockout']}" if old['knockout'] else ('Score above cut-off' if old['approved'] else 'Score below cut-off')
        st.markdown(f'<div class="card"><h4>Traditional bureau scorecard</h4>{pill}<br><br>'
                    f'<span class="big" style="color:{OLD}">{sc_txt}</span><br><span class="small">{why}</span></div>',
                    unsafe_allow_html=True)
    with c2:
        pill = ('<span class="pill approve">APPROVE</span>' if new['approved'] else '<span class="pill decline">DECLINE</span>')
        why = (f"Knock-out: {new['knockout']}" if new['knockout'] else
               f"Estimated default risk {new['pd']:.1%} · EMI Rs {d['emi']:,.0f}")
        thin = ' (thin-file launch cut-off)' if new['cut'] == 550 else ''
        st.markdown(f'<div class="card"><h4>Our model: broader signals</h4>{pill}<br><br>'
                    f'<span class="big" style="color:{ACCENT}">{new["score"]} / cut-off {new["cut"]}</span>'
                    f'<span class="small">{thin}</span><br><span class="small">{why}</span></div>', unsafe_allow_html=True)
    with c3:
        if new['approved']:
            items = ''.join(f"<li>{s['officer']} <span class='small'>(+{s['points']:.0f} pts)</span></li>" for s in new['strengths'])
            st.markdown(f'<div class="card"><h4>Top strengths</h4><ul style="margin:0;padding-left:18px">{items}</ul></div>',
                        unsafe_allow_html=True)
        else:
            items = ''.join(f"<li>{r['officer']} <span class='small'>({r['points']:.0f} pts)</span></li>" for r in new['reasons'])
            off = (f"<br><b>Counter-offer:</b> Rs {offer[0]:,.0f} (EMI Rs {offer[1]:,.0f})" if offer else
                   ('<br><b>Counter-offer:</b> none, decline stands' if not new['knockout'] else ''))
            st.markdown(f'<div class="card"><h4>Top reasons (officer view)</h4><ul style="margin:0;padding-left:18px">{items}</ul>{off}</div>',
                        unsafe_allow_html=True)

    st.markdown('')
    l, r = st.columns([1, 1.3])
    with l:
        st.markdown('**Message to the borrower**')
        st.markdown(f'<div class="letter">{S.borrower_message(new, offer)}</div>', unsafe_allow_html=True)
        st.caption('Plain-language reasons, as RBI\'s Fair Practices Code requires. Gender and location are never reasons.')
    with r:
        st.markdown('**How the score is built:** points vs an average applicant')
        p = new['points'].copy()
        p['Direction'] = np.where(p['vs average'] >= 0, 'Helps', 'Hurts')
        ch = alt.Chart(p).mark_bar().encode(
            y=alt.Y('Signal:N', sort='-x', title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X('vs average:Q', title='Points above / below an average applicant'),
            color=alt.Color('Direction:N', scale=alt.Scale(domain=['Helps', 'Hurts'], range=[GOOD, RED_C]), legend=None),
            tooltip=['Signal', 'Value', 'Band', 'Points', alt.Tooltip('vs average:Q', format='+.1f')]).properties(height=500)
        st.altair_chart(ch, use_container_width=True)
    with st.expander('Points table (new scorecard and traditional)'):
        x1, x2 = st.columns([1.5, 1])
        x1.dataframe(new['points'][['Signal', 'Value', 'Band', 'Points']], hide_index=True, use_container_width=True)
        x2.dataframe(old['points'], hide_index=True, use_container_width=True)

# =====================================================================================================  EARLY WARNING
with tab_ews:
    st.subheader('Early-warning portfolio monitor')
    st.caption('Every active 2024 loan is scored at each month-end. Red ≈ the riskiest 5% (visit within 7 days), Amber ≈ the next 10% (nudge).')
    months = sorted(ews.calendar_month.unique())
    m = st.select_slider('Month-end', options=months, value='2024-09' if '2024-09' in months else months[len(months) // 2])
    snap = ews[ews.calendar_month == m]
    cnt = snap.flag.value_counts()
    c = st.columns(5)
    for col, (name, color) in zip(c, [('Red', RED_C), ('Amber', AMBER_C), ('Green', GOOD),
                                      ('In collections (30+ DPD)', OLD)]):
        col.markdown(f'<div class="card"><h4>{name}</h4><span class="big" style="color:{color}">{cnt.get(name, 0):,}</span></div>',
                     unsafe_allow_html=True)
    c[4].markdown(f'<div class="card"><h4>Loans on book</h4><span class="big">{len(snap):,}</span></div>', unsafe_allow_html=True)
    st.markdown('')

    f1, f2, f3 = st.columns(3)
    fl = f1.multiselect('Flags', ['Red', 'Amber'], default=['Red', 'Amber'])
    oc = f2.multiselect('Occupation', sorted(snap.occupation.unique()))
    ar = f3.selectbox('Area', ['All', 'Rural', 'Urban'])
    w = snap[snap.flag.isin(fl)]
    if oc: w = w[w.occupation.isin(oc)]
    if ar != 'All': w = w[w.is_rural == (1 if ar == 'Rural' else 0)]
    w = w.sort_values('p', ascending=False)
    view = w[['borrower_id', 'occupation', 'month_on_book', 'p', 'flag', 'reason_1', 'reason_2', 'action']].rename(
        columns={'borrower_id': 'Loan', 'occupation': 'Occupation', 'month_on_book': 'Month on book',
                 'p': 'Risk (30+ DPD in 3m)', 'flag': 'Flag', 'reason_1': 'Main reason', 'reason_2': 'Second reason',
                 'action': 'Suggested action'})
    st.markdown(f'**Watch-list: {len(view):,} loans**')
    st.dataframe(view, hide_index=True, use_container_width=True, height=300,
                 column_config={'Risk (30+ DPD in 3m)': st.column_config.ProgressColumn(format='%.0f%%' if False else '%.2f', min_value=0, max_value=1)})

    st.markdown('**Loan history**')
    opts = view.Loan.tolist() or snap.borrower_id.tolist()
    dflt = ews[ews.borrower_id.isin(opts) & (ews.default_90dpd_12m == 1)].borrower_id
    ix = opts.index(dflt.iloc[0]) if len(dflt) else 0          # open a loan that later defaulted, to show the warning
    pick = st.selectbox('Loan', opts, index=ix, label_visibility='collapsed')
    h = ews[ews.borrower_id == pick].copy()
    h['Expected (seasonal normal)'] = h.avg_monthly_inflow_12m * h.season_index
    info = h.iloc[0]
    st.caption(f"{info.occupation} · {'rural' if info.is_rural else 'urban'} · EMI Rs {info.emi_due:,.0f} · "
               f"outcome: {'defaulted (90+ DPD)' if info.default_90dpd_12m else 'did not default'}")
    a1, a2 = st.columns(2)
    with a1:
        if h.monthly_inflow.notna().any():
            inc = h.melt(id_vars='calendar_month', value_vars=['monthly_inflow', 'Expected (seasonal normal)'],
                         var_name='Series', value_name='Rs')
            inc['Series'] = inc.Series.replace({'monthly_inflow': 'Actual bank/UPI inflow'})
            st.altair_chart(alt.Chart(inc).mark_line(point=True).encode(
                x=alt.X('calendar_month:N', title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Rs:Q', title='Monthly inflow (Rs)', scale=alt.Scale(zero=True)),
                color=alt.Color('Series:N', scale=alt.Scale(range=[ACCENT, OLD]), legend=alt.Legend(orient='bottom', title=None)),
                strokeDash=alt.StrokeDash('Series:N', legend=None)).properties(height=300, title='Income vs what is normal for the season'),
                use_container_width=True)
        else:
            st.info('No bank/UPI data for this borrower. The model relies on repayments, credit enquiries and udhaar records.')
    with a2:
        cmap = alt.Scale(domain=['Green', 'Amber', 'Red', 'In collections (30+ DPD)'], range=[GOOD, AMBER_C, RED_C, OLD])
        bars = alt.Chart(h).mark_bar().encode(
            x=alt.X('calendar_month:N', title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('p:Q', title='Risk score', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('flag:N', scale=cmap, legend=alt.Legend(orient='bottom', title=None)),
            tooltip=['calendar_month', 'flag', alt.Tooltip('p:Q', format='.2f'), 'payment_ratio', 'dpd', 'reason_1', 'action'])
        st.altair_chart(bars.properties(height=300, title='Monthly risk score and flag'), use_container_width=True)
    st.dataframe(h[['calendar_month', 'emi_due', 'amount_paid', 'dpd', 'new_external_loan_flag', 'udhaar_tab_change_pct',
                    'flag', 'reason_1', 'action']].rename(columns={'calendar_month': 'Month', 'emi_due': 'EMI due',
                    'amount_paid': 'Paid', 'dpd': 'DPD', 'new_external_loan_flag': 'New loan elsewhere',
                    'udhaar_tab_change_pct': 'Udhaar tab change %', 'flag': 'Flag', 'reason_1': 'Main reason',
                    'action': 'Action'}), hide_index=True, use_container_width=True)

# =====================================================================================================  HOW IT WORKS
with tab_how:
    st.subheader('How it works')
    a, b = st.columns(2)
    with a:
        st.markdown("""
**1. Approval scorecard (champion model)**
- Logistic regression on banded signals (Weight of Evidence), turned into **points**. It's transparent and auditable.
- **Signals:**
  - credit bureau (when available);
  - the loan officer's lean-month and average surplus estimates;
  - bank/UPI cash flows (volatility, bounces, savings);
  - electricity and mobile payments;
  - **Udhaar Passport** (verified informal repayment record);
  - SHG attendance and savings;
  - drought exposure.
- **Missing data is its own band**, so no-file applicants are scored on what they do have instead of being declined.
- **Decision:**
  - approve at 534 points;
  - 550 for thin-file applicants during launch;
  - policy knock-outs: 90+ DPD, FOIR above 50%.
- **Never used:** gender, state, location or segment labels.

**2. Explanations**
- The top 3 reasons are the signals furthest below an average applicant's points.
- Plain-language borrower message, with what would help.
- **Counter-offer:** the largest loan that passes the score, FOIR ≤ 50% and EMI ≤ average monthly surplus.
""")
    with b:
        st.markdown("""
**3. Early-warning model**
- Monthly logistic model predicting 30+ DPD in the next 3 months.
- **Signals:**
  - missed or partial payments;
  - **income vs the seasonal normal for the borrower's occupation**;
  - new loans and enquiries elsewhere;
  - a rising udhaar tab;
  - UPI activity;
  - drought-hit harvests.
- Red (≈5% of loans) → visit or call; Amber (≈10%) → nudge or check-in.
- Actions include priced repayment holidays, skip-vouchers and seasonal EMIs.

**4. Validation**
- Built on 2023 applicants, tested on Jan–Jun 2024 applicants (out of time).
- Baseline = traditional bureau scorecard with standard knock-outs (no file, 90+ DPD, FOIR above 50%), frozen before comparison.
- **Fairness:**
  - 80% rule on approval rates;
  - equal approval of good borrowers;
  - calibration by group;
  - proxy check.

**Limitations**
- Synthetic data; real-data validation needed.
- Thin-file risk drifted in 2024, so cut-offs need quarterly recalibration.
- Counter-offer performance needs a live pilot.
""")
