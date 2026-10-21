# Inclusive Credit Decisioning: prototype

EXL Live Project, IIM Lucknow · *Credit Assessment for Underserved Borrowers*
Team: S R Ashwin Narayan · Tejas Tijare · Utkarsh Kumar

A lender-facing web app for NBFCs and MFIs with four tabs:
1. **Results:** the traditional bureau scorecard vs our model on the 2024 test set (predictive power, fairness, value of each signal).
2. **Score an applicant:**
   - Pick any of the 3,319 test applicants, or type in your own.
   - Both models score the applicant side by side.
   - You get the top 3 reasons, a plain-language message to the borrower, a counter-offer, and a points breakdown.
3. **Early-warning portfolio:**
   - Month-by-month Red/Amber/Green watch-list with reasons and suggested actions.
   - Each loan's history shows income against its seasonal normal.
4. **How it works:** method and limitations.

All data is synthetic. The scoring logic in `scoring.py` reproduces the notebook models exactly; it was checked against all 3,319 test applicants.

## Run it on your laptop
```bash
pip install -r requirements.txt
streamlit run app.py
```
It opens at http://localhost:8501.

## Put it online for free (Streamlit Community Cloud)
1. **Create a repository:** create a free GitHub account, then make a new **public** repository (e.g. `exl-credit-prototype`).
2. **Upload the files:** use "Add file → Upload files" and drag in everything from this folder, keeping the `data/` folder:
   - `app.py`
   - `scoring.py`
   - `requirements.txt`
   - `README.md`
   - `data/`
3. **Connect Streamlit:** go to https://share.streamlit.io, sign in with GitHub, and click **Create app**.
4. **Point it at your code:** choose your repository and branch `main`, with main file `app.py`. Click **Deploy**.
5. **Share the link:** after 1–2 minutes you get a public link like `https://<name>.streamlit.app`. Put it in the report.

**Tip:** to open a specific applicant directly, add `?id=` to the link, for example `https://<name>.streamlit.app/?id=B00252` (a declined applicant who gets a counter-offer). Free apps go to sleep after a few days without visitors; open the link once before a presentation to wake it up.

## Files
| File | What it is |
|---|---|
| `app.py` | The web app |
| `scoring.py` | Baseline and new scorecard, reason codes, counter-offers |
| `data/config.json` | Cut-offs, bands and scorecard settings |
| `data/new_points.csv`, `data/baseline_points.csv` | Points tables from Steps 1–2 |
| `data/applicants_2024.csv` | 2024 test applicants with notebook scores |
| `data/ews_panel_2024.csv` | Monthly early-warning scores, flags, reasons and actions (Step 6) |
| `data/ews_params.json` | Early-warning model coefficients and thresholds |
| `data/step*.csv` | Results tables from Steps 3–7 |

To regenerate `data/` after changing a model, rerun the notebooks, then run `export_app_assets.py`.
