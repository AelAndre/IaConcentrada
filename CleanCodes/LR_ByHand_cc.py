#////////////////// Seoul Bikes - Linear Regression BY HAND //////////////////
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
#  Linear regression trained with batch gradient descent implemented without
#  framework. 
#
#/////////////////////////////////////////////////////////////////////////////

# ///// Libraries /////
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ///// Hyper parameters /////
ALFA = 0.05
MAX_EPOCHS = 300
SEED = 42

# ///// Error history /////
__errors__ = []
__errors_val__ = []


# ///// Paths /////
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'Data'
FIGURE_DIR = PROJECT_DIR / 'Figures'
TABLE_DIR = PROJECT_DIR / 'Tables'
RESULTS_DIR = PROJECT_DIR / 'Results'

for _d in (FIGURE_DIR, TABLE_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

# ///// Load Data /////
feature_names = list(pd.read_csv(DATA_DIR / 'X_train_seoul.csv').columns)

X_train = pd.read_csv(DATA_DIR / 'X_train_seoul.csv').values.tolist()
Y_train = pd.read_csv(DATA_DIR / 'Y_train_seoul.csv').values.flatten().tolist()

X_val = pd.read_csv(DATA_DIR / 'X_val_seoul.csv').values.tolist()
Y_val = pd.read_csv(DATA_DIR / 'Y_val_seoul.csv').values.flatten().tolist()

X_test = pd.read_csv(DATA_DIR / 'X_test_seoul.csv').values.tolist()
Y_test = pd.read_csv(DATA_DIR / 'Y_test_seoul.csv').values.flatten().tolist()


# ///// Add the bias slot to every sample /////
def add_bias(samples):
    for i in range(len(samples)):
        if isinstance(samples[i], list):
            samples[i] = [1] + samples[i]
        else:
            samples[i] = [1, samples[i]]
    return samples

X_train = add_bias(X_train)
X_val = add_bias(X_val)
X_test = add_bias(X_test)


# ///// Hypothesis /////
def h(params, sample):
    acum = 0
    for i in range(len(params)):
        acum = acum + params[i] * sample[i]
    return acum


# ///// Cost function (MSE) /////
def show_errors(params, samples, y, error_list=None):
    global __errors__
    if error_list is None:
        error_list = __errors__
    error_acum = 0
    for i in range(len(samples)):
        hyp = h(params, samples[i])
        error = hyp - y[i]
        error_acum = error_acum + error ** 2
    mean_error_param = error_acum / len(samples)
    error_list.append(mean_error_param)
    return mean_error_param


# ///// Gradient descent /////
def GD(params, samples, y, alfa):
    temp = list(params)
    for j in range(len(params)):
        acum = 0
        for i in range(len(samples)):
            error = h(params, samples[i]) - y[i]
            acum = acum + error * samples[i][j]
        temp[j] = params[j] - alfa * (1 / len(samples)) * acum
    return temp


# ///// Scaling (mean normalization, fitted on train only) /////
def scaling(samples, avgs=None, max_vals=None):
    samples = np.asarray(samples).T.tolist()
    calculate = avgs is None
    if calculate:
        avgs = [0] * len(samples)
        max_vals = [0] * len(samples)
    for i in range(1, len(samples)):          # index 0 is the bias column
        if calculate:
            acum = 0
            for j in range(len(samples[i])):
                acum = acum + samples[i][j]
            avgs[i] = acum / len(samples[i])
            max_vals[i] = max(samples[i])
            if max_vals[i] == 0:
                max_vals[i] = 1
        for j in range(len(samples[i])):
            samples[i][j] = (samples[i][j] - avgs[i]) / max_vals[i]
    samples = np.asarray(samples).T.tolist()
    return samples, avgs, max_vals

X_train_s, avgs, max_vals = scaling(X_train)
X_val_s, _, _ = scaling(X_val, avgs, max_vals)
X_test_s, _, _ = scaling(X_test, avgs, max_vals)


# ///// Initial weights /////
random.seed(SEED)
params = [random.uniform(-0.01, 0.01) for _ in range(len(X_train_s[0]))]


# ///// Train /////
epochs = 0
while True:
    oldparams = list(params)
    params = GD(params, X_train_s, Y_train, ALFA)
    show_errors(params, X_train_s, Y_train)
    show_errors(params, X_val_s, Y_val, __errors_val__)
    epochs = epochs + 1
    if (oldparams == params or epochs == MAX_EPOCHS):
        break


# ///// Prediction and metrics /////
def predict(samples, params):
    predictions = []
    for i in range(len(samples)):
        predictions.append(h(params, samples[i]))
    return predictions


def mse_manual(y_real, y_pred):
    error_acum = 0
    for i in range(len(y_real)):
        error_acum = error_acum + (y_pred[i] - y_real[i]) ** 2
    return error_acum / len(y_real)


def rmse_manual(y_real, y_pred):
    return mse_manual(y_real, y_pred) ** 0.5


def mae_manual(y_real, y_pred):
    error_acum = 0
    for i in range(len(y_real)):
        error_acum = error_acum + abs(y_pred[i] - y_real[i])
    return error_acum / len(y_real)


def r2_manual(y_real, y_pred):
    mean = sum(y_real) / len(y_real)
    ss_res = 0
    ss_tot = 0
    for i in range(len(y_real)):
        ss_res = ss_res + (y_real[i] - y_pred[i]) ** 2
        ss_tot = ss_tot + (y_real[i] - mean) ** 2
    return 1 - (ss_res / ss_tot)


def all_metrics(y_real, y_pred):
    return {"MSE": mse_manual(y_real, y_pred),
            "RMSE": rmse_manual(y_real, y_pred),
            "MAE": mae_manual(y_real, y_pred),
            "R2": r2_manual(y_real, y_pred)}

y_pred_val = predict(X_val_s, params)
y_pred_test = predict(X_test_s, params)

m_val = all_metrics(Y_val, y_pred_val)
m_test = all_metrics(Y_test, y_pred_test)


# ///// Output/////
print("=" * 62)
print("LINEAR REGRESSION BY HAND (batch gradient descent)")
print("=" * 62)
print(f"train: {len(X_train_s)} rows | val: {len(X_val_s)} rows | test: {len(X_test_s)} rows")
print(f"learning rate: {ALFA} | epochs run: {epochs}")
print(f"\nfinal MSE train: {__errors__[-1]:,.2f}")
print(f"final MSE val:   {__errors_val__[-1]:,.2f}")

weights_table = pd.DataFrame({
    "Parameter": ["Bias"] + feature_names,
    "Weight": params,
}).set_index("Parameter").round(4)

print("\nFINAL WEIGHTS")
print(weights_table.to_string())

mean_train = sum(Y_train)/len(Y_train)
baseline_test = [mean_train]*len(Y_test)
m_baseline = all_metrics(Y_test, baseline_test)

print(f"\nBASELINE (always predict the training mean of {mean_train:.2f})")
for k, v in m_baseline.items():
    print(f"  {k}: {v:.4f}")

print("\nVALIDATION")
for k, v in m_val.items():
    print(f"  {k}: {v:.4f}")
print("\nTEST")
for k, v in m_test.items():
    print(f"  {k}: {v:.4f}")


# ///// Figures /////
plt.figure(figsize=(8, 5))
plt.plot(__errors__, label="Train")
plt.plot(__errors_val__, label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Error (MSE)")
plt.title("Linear Regression by hand - Learning curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(FIGURE_DIR / "fig_byhand_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(Y_test, y_pred_test, alpha=0.5)
lims = [min(min(Y_test), min(y_pred_test)), max(max(Y_test), max(y_pred_test))]
plt.plot(lims, lims, 'k--')
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.title("Linear Regression by hand - Real vs Predicted (Test)")
plt.grid(True, alpha=0.3)
plt.savefig(FIGURE_DIR / "fig_byhand_real_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()


# ///// Export metrics for the comparison script /////
metrics_table = pd.DataFrame([
    {"Subset": "Validation", **m_val},
    {"Subset": "Test", **m_test},
]).set_index("Subset").round(4)

with open(TABLE_DIR / "table_byhand_metrics.tex", "w", encoding="utf-8") as f:
    f.write(metrics_table.to_latex(escape=True, float_format="%.4f"))

with open(TABLE_DIR / "table_byhand_weights.tex", "w", encoding="utf-8") as f:
    f.write(weights_table.to_latex(escape=True, float_format="%.4f"))

pd.DataFrame([
    {"Model": "Linear Regression (by hand)", "Subset": "val", **m_val},
    {"Model": "Linear Regression (by hand)", "Subset": "test", **m_test},
]).to_csv(RESULTS_DIR / "metrics_byhand.csv", index=False)

print("\nSaved: figures, table_byhand_metrics.tex, table_byhand_weights.tex, metrics_byhand.csv")
