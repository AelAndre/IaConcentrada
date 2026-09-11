#////////////// Seoul Bikes - Random Forest (FRAMEWORK: scikit-learn) /////////
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
#  Random forest regressor grabs a bunch of decision trees, each fitted on a
#  bootstrap sample of the training rows and considering a random subset of
#  features at every split. Averaging their predictions reduces the variance
#  that makes a single deep tree overfit, which is why depth is left
#  unrestricted here while the single tree needed a depth limit.
#
#  Like the single tree, splits use squared_error and no scaling is applied.
#
#  References:
#  Random Forest
#   https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#
#/////////////////////////////////////////////////////////////////////////////

# ///// Libraries /////
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ///// Configuration /////
N_ESTIMATORS = 200
SEED = 42

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

feature_names = list(X_train.columns)


# ///// Metrics helper /////
def metrics(y_true, y_pred):
    return {"MSE": mean_squared_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


# ///// Fit /////
rf = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                           random_state=SEED,
                           n_jobs=-1).fit(X_train, Y_train)

y_pred_train = rf.predict(X_train)
y_pred_val = rf.predict(X_val)
y_pred_test = rf.predict(X_test)

m_train = metrics(Y_train, y_pred_train)
m_val = metrics(Y_val, y_pred_val)
m_test = metrics(Y_test, y_pred_test)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)


# ///// Console output /////
print("=" * 62)
print("RANDOM FOREST REGRESSOR")
print("=" * 62)
print(f"n_estimators: {N_ESTIMATORS} | max_depth: unrestricted | random_state: {SEED}")

print("\nFEATURE IMPORTANCE")
print(importance.to_string(index=False))

print("\nTRAIN")
for k, v in m_train.items():
    print(f"  {k}: {v:.4f}")
print("\nVALIDATION")
for k, v in m_val.items():
    print(f"  {k}: {v:.4f}")
print("\nTEST")
for k, v in m_test.items():
    print(f"  {k}: {v:.4f}")
print("\nNote: R2 on train is much higher than on val/test because each tree is\n"
      "grown to full depth. The ensemble average is what keeps val/test usable.")


# ///// Figures /////
imp_asc = importance.sort_values("importance")
plt.figure(figsize=(8, 5))
plt.barh(imp_asc.feature, imp_asc.importance)
plt.xlabel("Importance")
plt.title("Random Forest - Feature importance")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fig_rf_importance.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(Y_test, y_pred_test, alpha=0.5)
lims = [min(Y_test.min(), y_pred_test.min()), max(Y_test.max(), y_pred_test.max())]
plt.plot(lims, lims, 'k--')
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.title("Random Forest - Real vs Predicted (Test)")
plt.grid(True, alpha=0.3)
plt.savefig(FIGURE_DIR / "fig_rf_real_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()


# ///// Export /////
with open(TABLE_DIR / "table_rf_importance.tex", "w") as f:
    f.write(importance.to_latex(escape=True, index=False, float_format="%.4f"))

pd.DataFrame([
    {"Model": "Random Forest", "Subset": "val", **m_val},
    {"Model": "Random Forest", "Subset": "test", **m_test},
]).to_csv(RESULTS_DIR / "metrics_rf.csv", index=False)

print("\nSaved: fig_rf_importance.png, fig_rf_real_vs_pred.png, "
      "table_rf_importance.tex, metrics_rf.csv")