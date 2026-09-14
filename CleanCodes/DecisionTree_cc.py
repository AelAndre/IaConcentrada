#////////////// Seoul Bikes - Decision Tree (FRAMEWORK: scikit-learn) /////////
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
#  Decision tree regressor. The splitting criterion is squared_error
#  not entropy because those are measures for classification trees. 
#  A regression tree chooses the threshold that most
#  reduces the variance of the target inside the resulting child nodes.
#  
#  max_depth is only for the validation subset only, test evaluation 
#  is un-touched until final evaluation.
#
#  References: 
#  Decision tree
#   https://scikit-learn.org/stable/modules/tree.html
#  Max depth
#   https://www.kaggle.com/code/pumalin/decision-trees-tutorial
#/////////////////////////////////////////////////////////////////////////////

# ///// Libraries /////
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

# ///// Configuration /////
DEPTHS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
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


# ///// Depth selection on validation /////
r2_train_curve, r2_val_curve = [], []
for d in DEPTHS:
    probe = DecisionTreeRegressor(max_depth=d, random_state=SEED).fit(X_train, Y_train)
    r2_train_curve.append(r2_score(Y_train, probe.predict(X_train)))
    r2_val_curve.append(r2_score(Y_val, probe.predict(X_val)))

best_depth = DEPTHS[int(np.argmax(r2_val_curve))]


# ///// Final tree /////
tree = DecisionTreeRegressor(max_depth=best_depth, random_state=SEED).fit(X_train, Y_train)

y_pred_val = tree.predict(X_val)
y_pred_test = tree.predict(X_test)

m_val = metrics(Y_val, y_pred_val)
m_test = metrics(Y_test, y_pred_test)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": tree.feature_importances_,
}).sort_values("importance", ascending=False)


# ///// Console output /////
print("=" * 62)
print("DECISION TREE REGRESSOR (criterion: squared_error)")
print("=" * 62)
print("Depth selection on the validation subset:")
print(f"{'depth':>6} | {'R2 train':>9} | {'R2 val':>9}")
for d, rt, rv in zip(DEPTHS, r2_train_curve, r2_val_curve):
    mark = "  <-- selected" if d == best_depth else ""
    print(f"{d:>6} | {rt:>9.4f} | {rv:>9.4f}{mark}")
print(f"\nSelected max_depth: {best_depth}")
print(f"Leaves in final tree: {tree.get_n_leaves()}")

print("\nFEATURE IMPORTANCE")
print(importance.to_string(index=False))

print("\nVALIDATION")
for k, v in m_val.items():
    print(f"  {k}: {v:.4f}")
print("\nTEST")
for k, v in m_test.items():
    print(f"  {k}: {v:.4f}")

print("\nFIRST SPLITS (top of the tree)")
print(export_text(tree, feature_names=feature_names, max_depth=2))


# ///// Figures /////
plt.figure(figsize=(8, 5))
plt.plot(DEPTHS, r2_train_curve, marker='o', label="Train")
plt.plot(DEPTHS, r2_val_curve, marker='o', label="Validation")
plt.axvline(best_depth, color='gray', linestyle='--')
plt.xlabel("max_depth")
plt.ylabel("R2")
plt.title("Decision Tree - depth vs performance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(FIGURE_DIR / "fig_tree_depth.png", dpi=150, bbox_inches="tight")
plt.close()

# Only the first levels are drawn: the full tree is unreadable on a page.
plt.figure(figsize=(22, 11))
plot_tree(tree, max_depth=3, feature_names=feature_names,
          filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree - first levels")
plt.savefig(FIGURE_DIR / "fig_tree_structure.png", dpi=110, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(Y_test, y_pred_test, alpha=0.5)
lims = [min(Y_test.min(), y_pred_test.min()), max(Y_test.max(), y_pred_test.max())]
plt.plot(lims, lims, 'k--')
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.title("Decision Tree - Real vs Predicted (Test)")
plt.grid(True, alpha=0.3)
plt.savefig(FIGURE_DIR / "fig_tree_real_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()


# ///// Export /////
with open(RESULTS_DIR / "tree_rules.txt", "w", encoding="utf-8") as f:
    f.write(export_text(tree, feature_names=feature_names, max_depth=3))

importance.to_csv(RESULTS_DIR / "importance_tree.csv", index=False)

pd.DataFrame([
    {"Model": "Decision Tree", "Subset": "val", **m_val},
    {"Model": "Decision Tree", "Subset": "test", **m_test},
]).to_csv(RESULTS_DIR / "metrics_tree.csv", index=False)

print("Saved: fig_tree_depth.png, fig_tree_structure.png, "
      "fig_tree_real_vs_pred.png, tree_rules.txt, metrics_tree.csv")