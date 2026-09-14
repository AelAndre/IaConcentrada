#////////////////// Seoul Bikes - Model comparison ////////////////////////////
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
#  Collects the metrics.csv written by each model script, adds the trivial
#  baseline (always predict the training mean) and prints the comparison table
#  plus the LaTeX version used in the report.
#
#/////////////////////////////////////////////////////////////////////////////

# ///// Libraries /////
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ///// Paths /////
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'Data'
FIGURE_DIR = PROJECT_DIR / 'Figures'
TABLE_DIR = PROJECT_DIR / 'Tables'
RESULTS_DIR = PROJECT_DIR / 'Results'

for _d in (FIGURE_DIR, TABLE_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

# ///// Baseline: always predict the training mean /////
Y_train = pd.read_csv(DATA_DIR / 'Y_train_seoul.csv').values.flatten()
Y_val = pd.read_csv(DATA_DIR / 'Y_val_seoul.csv').values.flatten()
Y_test = pd.read_csv(DATA_DIR / 'Y_test_seoul.csv').values.flatten()

mean_train = Y_train.mean()


def metrics(y_true, y_pred):
    return {"MSE": mean_squared_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}

rows = [
    {"Model": "Baseline (train mean)", "Subset": "val",
     **metrics(Y_val, np.full(len(Y_val), mean_train))},
    {"Model": "Baseline (train mean)", "Subset": "test",
     **metrics(Y_test, np.full(len(Y_test), mean_train))},
]

# ///// Collect every model that has been run /////
ORDER = ["metrics_byhand.csv", "metrics_ols.csv",
         "metrics_tree.csv", "metrics_rf.csv"]

missing = []
for fname in ORDER:
    fpath = RESULTS_DIR / fname
    if fpath.exists():
        rows.extend(pd.read_csv(fpath).to_dict("records"))
    else:
        missing.append(fname)

results = pd.DataFrame(rows)


# ///// Output /////
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(f"Training mean (baseline prediction): {mean_train:.2f} bikes\n")

print(results.round(4).to_string(index=False))

if missing:
    print(f"\nNot found (run those scripts first): {missing}")

test_only = (results[results.Subset == "test"]
             .set_index("Model")[["RMSE", "MAE", "R2"]]
             .round(3))

print("\nTEST SUBSET ONLY")
print(test_only.to_string())

best = test_only["R2"].idxmax()
print(f"\nBest model on test by R2: {best}")


# ///// Export /////
results.to_csv(RESULTS_DIR / "metrics_all_models.csv", index=False)

with open(TABLE_DIR / "table_model_comparison.tex", "w", encoding="utf-8") as f:
    f.write(test_only.to_latex(escape=True, float_format="%.3f"))

print("\nSaved: metrics_all_models.csv, table_model_comparison.tex")
