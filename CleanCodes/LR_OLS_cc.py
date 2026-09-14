#////////// Seoul Bikes - Linear Regression (FRAMEWORK: statsmodels OLS) //////
#  Evan Andre Santana Pacheco A01769493
#  TC3009C.601
#
#  Predictive Modeling of Urban Bike-Sharing Demand in Seoul
#    Seoul Bike Sharing Demand (UCI Machine Learning Repository)
#  https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand
#
#  Licence
#    Creative Commons Attribution 4.0 International (CC BY 4.0)
#
# Fitting Ordinary Least Squares (OLS) using statsmodels. We went with this 
# instead of scikit-learn's LinearRegression because it gives us 
# standard errors, t-statistics, p-values and confidence intervals for 
# every coefficient, for this scenario is more productive this way.
#
# We kept the features in their original units (no scaling). Since OLS 
# solves in closed form, it doesn't really need scaling, and leaving the 
# coefficients unscaled means we can directly interpret them as bikes 
# per unit of each feature.
#
#/////////////////////////////////////////////////////////////////////////////

# ///// Libraries /////
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ///// Paths /////
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'Data'
FIGURE_DIR = PROJECT_DIR / 'Figures'
TABLE_DIR = PROJECT_DIR / 'Tables'
RESULTS_DIR = PROJECT_DIR / 'Results'

for _d in (FIGURE_DIR, TABLE_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

# ///// Load Data /////
X_train = pd.read_csv(DATA_DIR / 'X_train_seoul.csv')
Y_train = pd.read_csv(DATA_DIR / 'Y_train_seoul.csv').values.flatten()

X_val = pd.read_csv(DATA_DIR / 'X_val_seoul.csv')
Y_val = pd.read_csv(DATA_DIR / 'Y_val_seoul.csv').values.flatten()

X_test = pd.read_csv(DATA_DIR / 'X_test_seoul.csv')
Y_test = pd.read_csv(DATA_DIR / 'Y_test_seoul.csv').values.flatten()


# ///// Metrics helper /////
def metrics(y_true, y_pred):
    return {"MSE": mean_squared_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


# ///// Fit OLS /////
X_train_sm = sm.add_constant(X_train)
X_val_sm = sm.add_constant(X_val, has_constant='add')
X_test_sm = sm.add_constant(X_test, has_constant='add')

ols = sm.OLS(Y_train, X_train_sm).fit()

y_pred_val = ols.predict(X_val_sm)
y_pred_test = ols.predict(X_test_sm)

m_val = metrics(Y_val, y_pred_val)
m_test = metrics(Y_test, y_pred_test)


# ///// Coefficient table with statistical significance /////
coef_table = pd.DataFrame({
    "Coefficient": ols.params,
    "Std. Error": ols.bse,
    "t": ols.tvalues,
    "P>|t|": ols.pvalues,
    "CI 2.5%": ols.conf_int()[0],
    "CI 97.5%": ols.conf_int()[1],
}).round(4)

not_significant = list(coef_table[coef_table["P>|t|"] >= 0.05].index)


# ///// Console output /////
print("=" * 62)
print("LINEAR REGRESSION - OLS (statsmodels)")
print("=" * 62)
print(ols.summary())

print("\nCOEFFICIENT TABLE")
print(coef_table.to_string())
print(f"\nNot significant at 5% (p >= 0.05): {not_significant if not_significant else 'none'}")
print(f"Durbin-Watson: {sm.stats.stattools.durbin_watson(ols.resid):.4f}"
      "   (2 = no autocorrelation; far below 2 means positive autocorrelation)")
print(f"Condition number: {ols.condition_number:,.1f}"
      "   (above ~1000 suggests multicollinearity)")

print("\nVALIDATION")
for k, v in m_val.items():
    print(f"  {k}: {v:.4f}")
print("\nTEST")
for k, v in m_test.items():
    print(f"  {k}: {v:.4f}")


# ///// Graph Real vs Predicted /////
plt.figure(figsize=(6, 6))
plt.scatter(Y_test, y_pred_test, alpha=0.5)
lims = [min(Y_test.min(), y_pred_test.min()), max(Y_test.max(), y_pred_test.max())]
plt.plot(lims, lims, 'k--')
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.title("OLS - Real vs Predicted (Test)")
plt.grid(True, alpha=0.3)
plt.savefig(FIGURE_DIR / "fig_ols_real_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()


# ///// Export /////
with open(RESULTS_DIR / "ols_summary.txt", "w", encoding="utf-8") as f:
    f.write(str(ols.summary()))

with open(TABLE_DIR / "table_ols_coefficients.tex", "w", encoding="utf-8") as f:
    f.write(coef_table.to_latex(escape=True, float_format="%.4f"))

pd.DataFrame([
    {"Model": "Linear Regression (OLS)", "Subset": "val", **m_val},
    {"Model": "Linear Regression (OLS)", "Subset": "test", **m_test},
]).to_csv(RESULTS_DIR / "metrics_ols.csv", index=False)

print("\nSaved: fig_ols_real_vs_pred.png, ols_summary.txt, "
      "table_ols_coefficients.tex, metrics_ols.csv")
