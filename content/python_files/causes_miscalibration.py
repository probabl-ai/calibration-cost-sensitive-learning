# %%
import numpy as np


def xor_generator(n_samples=1_000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(low=-3, high=3, size=(n_samples, 2))
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    return X, y


# %%
import matplotlib.pyplot as plt

X, y = xor_generator(seed=0)
_, ax = plt.subplots()
ax.scatter(*X.T, c=y, cmap="coolwarm", alpha=0.5)
ax.set(
    xlim=(-3, 3),
    ylim=(-3, 3),
    xlabel="Feature 1",
    ylabel="Feature 2",
    title="XOR problem",
    aspect="equal",
)

# %%
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    SplineTransformer(n_knots=20),
    PolynomialFeatures(degree=2, interaction_only=True),
    LogisticRegression(),
)
model.fit(X, y)

# %%
from sklearn.inspection import DecisionBoundaryDisplay

_, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(
    model, X, ax=ax, cmap="coolwarm", response_method="predict_proba"
)
ax.scatter(*X.T, c=y, cmap="coolwarm", alpha=0.5)
ax.set(
    xlim=(-3, 3),
    ylim=(-3, 3),
    xlabel="Feature 1",
    ylabel="Feature 2",
    title="XOR problem",
    aspect="equal",
)

# %%
from sklearn.calibration import CalibrationDisplay

CalibrationDisplay.from_estimator(model, X, y, strategy="quantile", n_bins=10)

# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=20_000,
    n_features=2,
    n_redundant=0,
    weights=[0.1, 0.9],
    class_sep=1,
    random_state=1,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# %%
model = LogisticRegression()
model.fit(X_train, y_train)

# %%
from sklearn.metrics import classification_report

print(classification_report(y_test, model.predict(X_test)))

# %%
_, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(
    model,
    X_test,
    ax=ax,
    cmap="coolwarm",
    response_method="predict",
    alpha=0.5,
)
ax.scatter(*X_test.T, c=y_test, cmap="coolwarm", alpha=0.5)
ax.set(xlabel="Feature 1", ylabel="Feature 2")

# %%
model.set_params(class_weight="balanced").fit(X_train, y_train)

# %%
from sklearn.metrics import classification_report

print(classification_report(y_test, model.predict(X_test)))

# %%
_, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(
    model,
    X_test,
    ax=ax,
    cmap="coolwarm",
    response_method="predict",
    alpha=0.5,
)
ax.scatter(*X_test.T, c=y_test, cmap="coolwarm", alpha=0.5)
ax.set(xlabel="Feature 1", ylabel="Feature 2")

# %%
model_vanilla = LogisticRegression().fit(X_train, y_train)
model_reweighted = LogisticRegression(class_weight="balanced").fit(X_train, y_train)

# %%
disp = CalibrationDisplay.from_estimator(
    model_vanilla, X_test, y_test, strategy="quantile"
)
CalibrationDisplay.from_estimator(
    model_reweighted, X_test, y_test, strategy="quantile", ax=disp.ax_
)

# %%
