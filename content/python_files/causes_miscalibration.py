# %% [markdown]
#
# # The causes of miscalibration
#
# ## Effect of under-fitting and over-fitting on model calibration
#
# In this section, we look at the effect of under-fitting and over-fitting on the
# calibration of a model.
#
# Let's start by defining our classification problem: we use the so-called XOR problem.
# The function `xor_generator` generates a dataset with two features and the target
# variable following the XOR logic. We add some noise to the generative process.

# %%
import numpy as np


def xor_generator(n_samples=1_000, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.uniform(low=-3, high=3, size=(n_samples, 2))
    unobserved = rng.normal(loc=0, scale=0.5, size=(n_samples, 2))
    y = np.logical_xor(X[:, 0] + unobserved[:, 0] > 0, X[:, 1] + unobserved[:, 1] > 0)
    return X, y


# %% [markdown]
#
# We can now generate a dataset and visualize it.


# %%
import matplotlib.pyplot as plt

X_train, y_train = xor_generator(seed=0)
_, ax = plt.subplots()
ax.scatter(*X_train.T, c=y_train, cmap="coolwarm", edgecolors="black", alpha=0.5)
_ = ax.set(
    xlim=(-3, 3),
    ylim=(-3, 3),
    xlabel="Feature 1",
    ylabel="Feature 2",
    title="XOR problem",
    aspect="equal",
)

# %% [markdown]
#
# The XOR problem exhibits a non-linear decision link between the features and the
# the target variable. Therefore, a linear model will not be able to separate the
# classes correctly. Let's confirm this intuition by fitting a logistic regression
# model to such dataset.

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# %% [markdown]
#
# To check the decision boundary of the model, we will use an independent test set.

# %%
from sklearn.inspection import DecisionBoundaryDisplay

X_test, y_test = xor_generator(n_samples=1_000, seed=1)

fig, ax = plt.subplots()
params = {
    "cmap": "coolwarm",
    "response_method": "predict_proba",
    "plot_method": "pcolormesh",
    # make sure to have a range of 0 to 1 for the probability
    "vmin": 0,
    "vmax": 1,
}
disp = DecisionBoundaryDisplay.from_estimator(model, X_test, ax=ax, **params)
ax.scatter(*X_test.T, c=y_test, cmap=params["cmap"], edgecolors="black", alpha=0.5)
fig.colorbar(disp.surface_, ax=ax, label="Probability estimate")
_ = ax.set(
    xlim=(-3, 3),
    ylim=(-3, 3),
    xlabel="Feature 1",
    ylabel="Feature 2",
    title="Soft decision boundary of a logistic regression",
    aspect="equal",
)

# %% [markdown]
#
# We see that the probability estimates is almost constant and the model is really
# uncertain with an estimated probability of 0.5 for all samples in the test set.
#
# We therefore need a more expressive model to capture the non-linear relationship
# between the features and the target variable. Crafting a pre-processing step to
# transform the features into a higher-dimensional space could help. We create a
# pipeline that includes a spline transformation and a polynomial transformation before
# to train our logistic regression model.

# %%
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    SplineTransformer(),
    PolynomialFeatures(),
    LogisticRegression(),
)
model.fit(X_train, y_train)

# %% [markdown]
#
# Let's check the decision boundary of the model on the test set.

# %%
fig, ax = plt.subplots()
disp = DecisionBoundaryDisplay.from_estimator(model, X_test, ax=ax, **params)
ax.scatter(*X_test.T, c=y_test, cmap=params["cmap"], edgecolors="black", alpha=0.5)
fig.colorbar(disp.surface_, ax=ax, label="Probability estimate")
_ = ax.set(
    xlim=(-3, 3),
    ylim=(-3, 3),
    xlabel="Feature 1",
    ylabel="Feature 2",
    title="Soft decision boundary of a logistic regression\n with pre-processing",
    aspect="equal",
)

# %% [markdown]
#
# We see that our model is capable of capturing the non-linear relationship between
# the features and the target variable. The probability estimates are now varying
# across the samples. We could check the calibration of our model using the calibration
# curve.

# %%
from sklearn.calibration import CalibrationDisplay

disp = CalibrationDisplay.from_estimator(
    model,
    X_test,
    y_test,
    strategy="quantile",
    n_bins=10,
)
_ = disp.ax_.set(aspect="equal")

# %% [markdown]
#
# We observe that the calibration of the model is not perfect. So is there a way to
# improve the calibration of our model?
#
# As an exercise, you could try to:
# - modify the parameter `n_knots` of the `SplineTransformer`,
# - modify the parameter `degree` of the `PolynomialFeatures`,
# - modify the parameter `interaction_only` of the `PolynomialFeatures`,
# - modify the parameter `C` of the `LogisticRegression`.
#
# The idea is to observe the effect in terms of under-/over-fitting by looking at the
# decision boundary display and the effect on the model calibration on the calibration
# curve.

# %%
import pprint
from sklearn.model_selection import ParameterGrid

param_grid = list(
    ParameterGrid(
        {
            "logisticregression__C": np.logspace(-3, 3, 7),
            "splinetransformer__n_knots": [5, 10],
            "polynomialfeatures__degree": [2, 3],
            "polynomialfeatures__interaction_only": [True, False],
        }
    )
)

boundary_figure, boundary_axes = plt.subplots(
    nrows=8, ncols=7, figsize=(50, 60), sharex=True, sharey=True
)
calibration_figure, calibration_axes = plt.subplots(
    nrows=8, ncols=7, figsize=(50, 60), sharex=True, sharey=True
)
params["plot_method"] = "contourf"

pp = pprint.PrettyPrinter(indent=1, width=1)
for idx, (model_params, ax_boundary, ax_calibration) in enumerate(zip(
    param_grid, boundary_axes.ravel(), calibration_axes.ravel()
)):
    model.set_params(**model_params).fit(X_train, y_train)
    # Display the results
    disp = DecisionBoundaryDisplay.from_estimator(
        model, X_test, ax=ax_boundary, **params
    )
    ax_boundary.scatter(
        *X_test.T, c=y_test, cmap=params["cmap"], edgecolor="black", alpha=0.5
    )
    ax_boundary.set(
        xlim=(-3, 3),
        ylim=(-3, 3),
        aspect="equal",
        title=f"{pp.pformat(model_params)}",
    )

    CalibrationDisplay.from_estimator(
        model,
        X_test,
        y_test,
        strategy="quantile",
        n_bins=10,
        ax=ax_calibration,
    )
    ax_calibration.set(aspect="equal", title=f"{pp.pformat(model_params)}")

# %%
#
# ## Effect of resampling on model calibration

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
