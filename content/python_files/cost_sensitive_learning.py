# %% [markdown]
#
# # Cost-sensitive learning to optimize business metrics
#
# As stated in the introduction of this tutorial, many real-world applications are
# interested by taking operating decisions. A predictive model under such
# circumstances should optimized a "utility function" or also called "business
# metric". The aim is therefore to maximize a gain or minimize a cost that is
# related to the decision taken by the model.
#
# In this tutorial, we describe a concrete example based on the credit card
# fraud detection problem. We first describe the dataset to train our predictive model
# and the data used to evaluate the operating decisions in this application. Then, we
# show a couple of approaches, each having different requirements, to get the optimal
# predictive model.
#
# ## The credit card dataset
#
# The problem that we solve in this tutorial is to detect fraudulent credit card
# transactions. The dataset is available on OpenML

# %%
# Explicit import pyarrow since this is an optional dependency of pandas to trigger
# the fetching when using JupyterLite with pyodide kernel.
# Note this is an unnecessary import if you are not using JupyterLite with pyodide.
import pyarrow  # noqa: F401
import pandas as pd

credit_card = pd.read_parquet("../datasets/credit_card.parquet", engine="pyarrow")
credit_card.info()

# %%
# The dataset contains information about credit card records from which some are
# fraudulent and others are legitimate. The goal is therefore to predict whether or
# not a credit card record is fraudulent.
columns_to_drop = ["Class"]
data = credit_card.frame.drop(columns=columns_to_drop)
target = credit_card.frame["Class"].astype(int)

# %%
# First, we check the class distribution of the datasets.
target.value_counts(normalize=True)

# %%
# The dataset is highly imbalanced with fraudulent transaction representing only 0.17%
# of the data. Since we are interested in training a machine learning model, we should
# also make sure that we have enough samples in the minority class to train the model.
target.value_counts()

# %%
# We observe that we have around 500 samples that is on the low end of the number of
# samples required to train a machine learning model. In addition of the target
# distribution, we check the distribution of the amount of the
# fraudulent transactions.
import matplotlib.pyplot as plt

fraud = target == 1
amount_fraud = data["Amount"][fraud]
_, ax = plt.subplots()
ax.hist(amount_fraud, bins=30)
ax.set_title("Amount of fraud transaction")
_ = ax.set_xlabel("Amount (€)")

# %%
# Addressing the problem with a business metric
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now, we create the business metric that depends on the amount of each transaction. We
# define the cost matrix similarly to [2]_. Accepting a legitimate transaction provides
# a gain of 2% of the amount of the transaction. However, accepting a fraudulent
# transaction result in a loss of the amount of the transaction. As stated in [2]_, the
# gain and loss related to refusals (of fraudulent and legitimate transactions) are not
# trivial to define. Here, we define that a refusal of a legitimate transaction
# is estimated to a loss of 5€ while the refusal of a fraudulent transaction is
# estimated to a gain of 50€. Therefore, we define the following function to
# compute the total benefit of a given decision:


def business_metric(y_true, y_pred, amount):
    mask_true_positive = (y_true == 1) & (y_pred == 1)
    mask_true_negative = (y_true == 0) & (y_pred == 0)
    mask_false_positive = (y_true == 0) & (y_pred == 1)
    mask_false_negative = (y_true == 1) & (y_pred == 0)
    fraudulent_refuse = mask_true_positive.sum() * 50
    fraudulent_accept = -amount[mask_false_negative].sum()
    legitimate_refuse = mask_false_positive.sum() * -5
    legitimate_accept = (amount[mask_true_negative] * 0.02).sum()
    return fraudulent_refuse + fraudulent_accept + legitimate_refuse + legitimate_accept


# %%
# From this business metric, we create a scikit-learn scorer that given a fitted
# classifier and a test set compute the business metric. In this regard, we use
# the :func:`~sklearn.metrics.make_scorer` factory. The variable `amount` is an
# additional metadata to be passed to the scorer and we need to use
# :ref:`metadata routing <metadata_routing>` to take into account this information.
import sklearn
from sklearn.metrics import make_scorer

sklearn.set_config(enable_metadata_routing=True)
business_scorer = make_scorer(business_metric).set_score_request(amount=True)

# %%
# So at this stage, we observe that the amount of the transaction is used twice: once
# as a feature to train our predictive model and once as a metadata to compute the
# the business metric and thus the statistical performance of our model. When used as a
# feature, we are only required to have a column in `data` that contains the amount of
# each transaction. To use this information as metadata, we need to have an external
# variable that we can pass to the scorer or the model that internally routes this
# metadata to the scorer. So let's create this variable.
amount = credit_card.frame["Amount"].to_numpy()

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test, amount_train, amount_test = (
    train_test_split(
        data, target, amount, stratify=target, test_size=0.5, random_state=42
    )
)

# %%
# We first evaluate some baseline policies to serve as reference. Recall that
# class "0" is the legitimate class and class "1" is the fraudulent class.
from sklearn.dummy import DummyClassifier

always_accept_policy = DummyClassifier(strategy="constant", constant=0)
always_accept_policy.fit(data_train, target_train)
benefit = business_scorer(
    always_accept_policy, data_test, target_test, amount=amount_test
)
print(f"Benefit of the 'always accept' policy: {benefit:,.2f}€")

# %%
# A policy that considers all transactions as legitimate would create a profit of
# around 220,000€. We make the same evaluation for a classifier that predicts all
# transactions as fraudulent.
always_reject_policy = DummyClassifier(strategy="constant", constant=1)
always_reject_policy.fit(data_train, target_train)
benefit = business_scorer(
    always_reject_policy, data_test, target_test, amount=amount_test
)
print(f"Benefit of the 'always reject' policy: {benefit:,.2f}€")


# %%
# Such a policy would entail a catastrophic loss: around 670,000€. This is
# expected since the vast majority of the transactions are legitimate and the
# policy would refuse them at a non-trivial cost.
#
# A predictive model that adapts the accept/reject decisions on a per
# transaction basis should ideally allow us to make a profit larger than the
# 220,000€ of the best of our constant baseline policies.
#
# We start with a logistic regression model with the default decision threshold
# at 0.5. Here we tune the hyperparameter `C` of the logistic regression with a
# proper scoring rule (the log loss) to ensure that the model's probabilistic
# predictions returned by its `predict_proba` method are as accurate as
# possible, irrespectively of the choice of the value of the decision
# threshold.
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logistic_regression = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {"logisticregression__C": np.logspace(-6, 6, 13)}
model = GridSearchCV(logistic_regression, param_grid, scoring="neg_log_loss").fit(
    data_train, target_train
)
model

# %%
print(
    "Benefit of logistic regression with default threshold: "
    f"{business_scorer(model, data_test, target_test, amount=amount_test):,.2f}€"
)

# %%
# The business metric shows that our predictive model with a default decision
# threshold is already winning over the baseline in terms of profit and it would be
# already beneficial to use it to accept or reject transactions instead of
# accepting all transactions.
#
# Tuning the decision threshold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now the question is: is our model optimum for the type of decision that we want to do?
# Up to now, we did not optimize the decision threshold. We use the
# :class:`~sklearn.model_selection.TunedThresholdClassifierCV` to optimize the decision
# given our business scorer. To avoid a nested cross-validation, we will use the
# best estimator found during the previous grid-search.
from sklearn.model_selection import TunedThresholdClassifierCV

tuned_model = TunedThresholdClassifierCV(
    estimator=model.best_estimator_,
    scoring=business_scorer,
    thresholds=100,
    n_jobs=2,
)

# %%
# Since our business scorer requires the amount of each transaction, we need to pass
# this information in the `fit` method. The
# :class:`~sklearn.model_selection.TunedThresholdClassifierCV` is in charge of
# automatically dispatching this metadata to the underlying scorer.
tuned_model.fit(data_train, target_train, amount=amount_train)

# %%
# We observe that the tuned decision threshold is far away from the default 0.5:
print(f"Tuned decision threshold: {tuned_model.best_threshold_:.2f}")

# %%
print(
    "Benefit of logistic regression with a tuned threshold: "
    f"{business_scorer(tuned_model, data_test, target_test, amount=amount_test):,.2f}€"
)

# %%
# We observe that tuning the decision threshold increases the expected profit
# when deploying our model - as indicated by the business metric. It is therefore
# valuable, whenever possible, to optimize the decision threshold with respect
# to the business metric.
#
# Manually setting the decision threshold instead of tuning it
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the previous example, we used the
# :class:`~sklearn.model_selection.TunedThresholdClassifierCV` to find the optimal
# decision threshold. However, in some cases, we might have some prior knowledge about
# the problem at hand and we might be happy to set the decision threshold manually.
#
# The class :class:`~sklearn.model_selection.FixedThresholdClassifier` allows us to
# manually set the decision threshold. At prediction time, it behave as the previous
# tuned model but no search is performed during the fitting process.
#
# Here, we will reuse the decision threshold found in the previous section to create a
# new model and check that it gives the same results.
from sklearn.model_selection import FixedThresholdClassifier

model_fixed_threshold = FixedThresholdClassifier(
    estimator=model, threshold=tuned_model.best_threshold_
).fit(data_train, target_train)

# %%
business_score = business_scorer(
    model_fixed_threshold, data_test, target_test, amount=amount_test
)
print(f"Benefit of logistic regression with a tuned threshold:  {business_score:,.2f}€")

# %%
# We observe that we obtained the exact same results but the fitting process
# was much faster since we did not perform any hyper-parameter search.
#
# Finally, the estimate of the (average) business metric itself can be unreliable, in
# particular when the number of data points in the minority class is very small.
# Any business impact estimated by cross-validation of a business metric on
# historical data (offline evaluation) should ideally be confirmed by A/B testing
# on live data (online evaluation). Note however that A/B testing models is
# beyond the scope of the scikit-learn library itself.

# %%
# Fixed Elkan-optimal threshold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Under the assumption that the probabilistic classifier is well-calibrated.


# %%
def elkan_optimal_threshold(amount):
    # Cost matrix (negative of gain matrix)
    c00 = -0.02 * amount  # Accepting a legitimate transaction
    c01 = amount  # Accepting a fraudulent transaction
    c10 = 5  # Refusing a legitimate transaction
    c11 = -50  # Refusing a fraudulent transaction
    optimal_threshold = (c10 - c00) / (c10 - c00 + c01 - c11)
    return optimal_threshold


elkan_optimal_threshold(amount_test).mean()

# %%
elkan_optimal_threshold(amount_test[amount_test > 100]).mean()

# %%
elkan_optimal_threshold(amount_test[amount_test <= 100]).mean()

# %%
_ = plt.hist(elkan_optimal_threshold(amount_test), bins=30)
# %%
import matplotlib.pyplot as plt

x = np.linspace(amount_test.min(), amount_test.max(), 1000)
plt.plot(x, elkan_optimal_threshold(x))
plt.xlabel("Amount (€)")
_ = plt.ylabel("Optimal threshold")

# %%
fixed_elkan_model = FixedThresholdClassifier(
    estimator=model.best_estimator_,
    threshold=elkan_optimal_threshold(amount_test).mean(),
).fit(data_train, target_train)

business_score = business_scorer(
    fixed_elkan_model, data_test, target_test, amount=amount_test
)
print(
    f"Benefit of logistic regression with a fixed mean theoretical threshold: "
    f"{business_score:,.2f}€"
)


# %%
from sklearn.calibration import CalibrationDisplay

disp = CalibrationDisplay.from_estimator(
    model.best_estimator_, data_test, target_test, strategy="quantile", n_bins=5
)
_ = disp.ax_.set(xlim=(1e-7, 0.03), ylim=(1e-7, 0.03), xscale="log", yscale="log")

# %%
from sklearn.calibration import CalibratedClassifierCV

calibrated_estimator = CalibratedClassifierCV(
    model.best_estimator_, method="isotonic"
).fit(data_train, target_train)
disp = CalibrationDisplay.from_estimator(
    calibrated_estimator, data_test, target_test, strategy="quantile", n_bins=5
)
_ = disp.ax_.set(xlim=(1e-7, 0.03), ylim=(1e-7, 0.03), xscale="log", yscale="log")

# %%
fixed_elkan_model = FixedThresholdClassifier(
    estimator=calibrated_estimator,
    threshold=elkan_optimal_threshold(amount_test).mean(),
).fit(data_train, target_train)

business_score = business_scorer(
    fixed_elkan_model, data_test, target_test, amount=amount_test
)
print(
    f"Benefit of recalibrated logistic regression with a fixed mean theoretical "
    f" threshold:  {business_score:,.2f}€"
)


# %%
# Variable Elkan-optimal threshold - predict-time optimization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Also under the assumption that the probabilistic classifier is well-calibrated.


class VariableThresholdClassifier:

    def __init__(self, estimator, variable_threshold):
        self.estimator = estimator
        self.variable_threshold = variable_threshold

    def fit(self, X, y):
        return self

    def predict(self, X, amount):
        proba = self.estimator.predict_proba(X)[:, 1]
        return (proba >= self.variable_threshold(amount)).astype(int)


business_score = business_metric(
    target_test,
    VariableThresholdClassifier(
        model.best_estimator_,
        variable_threshold=elkan_optimal_threshold,
    ).predict(data_test, amount=amount_test),
    amount=amount_test,
)
print(
    f"Benefit of logistic regression with optimal variable threshold: "
    f"{business_score:,.2f}€"
)

# %%
business_score = business_metric(
    target_test,
    VariableThresholdClassifier(
        calibrated_estimator,
        variable_threshold=elkan_optimal_threshold,
    ).predict(data_test, amount=amount_test),
    amount=amount_test,
)
print(
    f"Benefit of recalibrated logistic regression with optimal variable threshold: "
    f"{business_score:,.2f}€"
)

# %%
business_score = business_metric(
    target_test,
    target_test,
    amount=amount_test,
)
print(f"Benefit of oracle decisions (not reachable):  {business_score:,.2f}€")

# %%
