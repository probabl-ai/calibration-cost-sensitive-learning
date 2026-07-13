# %%
from sklearn.datasets import fetch_openml

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
target_count = y.value_counts()

# %%
from imblearn.datasets import make_imbalance

minority_class = ">50K"
X, y = make_imbalance(
    X,
    y,
    sampling_strategy={minority_class: int(target_count[minority_class] / 10)},
    random_state=0,
)

# %%
y.value_counts()

# %%
from skrub import tabular_pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

random_forest = tabular_pipeline(
    RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=0)
)
balanced_random_forest = tabular_pipeline(
    BalancedRandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=0)
)

# %%
import numpy as np
from sklearn.model_selection import GridSearchCV

# FIXME: use roc auc to decouple calibration vs. ranking power for both models

random_forest_gs = GridSearchCV(
    random_forest,
    param_grid={"randomforestclassifier__max_leaf_nodes": np.arange(10, 2_000, 15)},
    scoring="neg_log_loss",
)

# %%
balanced_random_forest_gs = GridSearchCV(
    balanced_random_forest,
    param_grid={
        "balancedrandomforestclassifier__max_leaf_nodes": np.arange(10, 2_000, 15)
    },
    scoring="neg_log_loss",
)

# %%
# from skore import CrossValidationReport
# import joblib

# report_rf = CrossValidationReport(random_forest_gs, X, y)
# joblib.dump(report_rf, "report_rf.joblib")

# %%
# report_brf = CrossValidationReport(balanced_random_forest_gs, X, y)
# joblib.dump(report_brf, "report_brf.joblib")

# %%
import joblib

report_rf = joblib.load("report_rf.joblib")
report_brf = joblib.load("report_brf.joblib")

# %%
from skore import ComparisonReport

comparison_report = ComparisonReport(
    {"Random Forest": report_rf, "Balanced Random Forest": report_brf}
)

# %%
summary = comparison_report.metrics.summarize(
    metric=["roc_auc", "log_loss", "brier_score", "fit_time", "predict_time"]
)
summary.frame()

# %%
plot_kwargs = {
    "palette": {"Random Forest":"tab:blue", "Balanced Random Forest":"tab:orange"}
    }
display = comparison_report.metrics.roc()
display.set_style(relplot_kwargs=plot_kwargs)
display.plot(label=">50K", subplot_by=None)

# %%
display = comparison_report.metrics.precision_recall()
display.set_style(relplot_kwargs=plot_kwargs)
display.plot(label=">50K", subplot_by=None)

# %%
# TODO: Recalibrate using the close form based on the true target
# TODO: look at the `n_estimators` effects