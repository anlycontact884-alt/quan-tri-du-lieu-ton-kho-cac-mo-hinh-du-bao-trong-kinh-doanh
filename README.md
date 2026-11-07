# Inventory Decision Support System

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Place `sales_data.csv` in `data/sales_data.csv`, or upload it from the sidebar.

Required columns: `Date`, `Store ID`, `Product ID`, `Demand`.  
Optional column: `Inventory Level`.

## Pages
1. Executive Overview
2. Demand Forecasting
3. Inventory Monitoring
4. Replenishment Decision
5. Agentic AI Decision Center
6. Inventory Policy Evaluation
7. Model & System Monitoring

## Methodological boundaries
- Forecasting page currently displays a transparent Seasonal Naive lag-7 baseline, not the trained Random Forest Global or quantile models from the research notebook.
- ROP is computed from empirical lead-time demand by Store–Product; lead time and service level are adjustable assumptions.
- Inventory Level is taken from the uploaded dataset. Missing values are treated as zero for replenishment calculations and should be reviewed.
- Agent output is rule-based and requires human review. It does not submit purchase orders or persist approval actions.
- Policy comparison reads `artifacts/policy_comparison.csv` exported from the notebook.

## Optional notebook exports
`artifacts/model_metrics.csv` columns: `Model,MAE,RMSE,WAPE`  
`artifacts/policy_comparison.csv` columns: `Policy,Stockout Rate,Fill Rate,Mean Inventory,Orders,Stockout Units`

## Deploy with GitHub + Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and folders `data/`, `artifacts/`.
3. Add `sales_data.csv` to `data/` only if the dataset license permits public redistribution. Otherwise use the in-app uploader; keep raw data out of the public repository.
4. Open Streamlit Community Cloud → Create app.
5. Select the repository, branch `main`, and main file `app.py`; deploy.
