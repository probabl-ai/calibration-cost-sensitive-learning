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
# transactions. The dataset is available on OpenML at the following URL:
# https://openml.org/search?type=data&sort=runs&status=active&id=1597
#
# We have a local copy of the dataset in the `datasets` folder. Let's load the dataset
# and check the data that we have at hand.

# %%

# Explicitly import pyarrow since it is an optional dependency of pandas to trigger
# the fetching when using JupyterLite with pyodide kernel.
# Note this is an unnecessary import if you are not using JupyterLite with pyodide.
import pyarrow  # noqa: F401
import pandas as pd

credit_card = pd.read_parquet("../datasets/credit_card.parquet", engine="pyarrow")
credit_card.info()

# %% [markdown]
#
# The target column is the "Class" column. It informs us whether a transaction
# is fraudulent (class 1) or legitimate (class 0).
#
# We see a set of features that are anonymized starting with "V". Looking at the dataset
# description in OpenML, we learn that those features are the result of a PCA
# transformation. The only non-transformed features are the "Time" and "Amount" columns.
# The "Time" corresponds to the number of seconds elapsed between this transaction and
# the first transaction in the dataset. The "Amount" is the amount of the transaction.
#
# We first split the target from the data that we want to use to train our predictive
# model.

# %%
target_name = "Class"
data = credit_card.drop(columns=target_name)
target = credit_card[target_name].astype(int)

# %% [markdown]
#
# The credit card fraud detection problem also has a special characteristic: the
# dataset is highly imbalanced. We can check the distribution of the target to
# confirm this.

# %%
target.value_counts(normalize=True)

# %% [markdown]
#
# The dataset is highly imbalanced with fraudulent transaction representing only 0.17%
# of the data. Since we are interested in training a machine learning model, we should
# also make sure that we have enough samples in the minority class to train the model
# by looking at the absolute number of samples.

# %%
target.value_counts()

# %% [markdown]
#
# We observe that we have around 500 samples that is on the low end of the number of
# samples required to train a machine learning model. In addition of the target
# distribution, we check the distribution of the amount of the legitimate and fraudulent
# separately transactions.

# %%
import numpy as np
import matplotlib.pyplot as plt

amount_groupby_class = pd.concat([data["Amount"], target], axis=1).groupby("Class")[
    "Amount"
]

_, ax = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)
bins = np.linspace(0, data["Amount"].max(), 30)
for class_id, amount in amount_groupby_class:
    ax[class_id].hist(amount, bins=bins, edgecolor="black", density=True)
    ax[class_id].set(
        xlabel="Amount (€)",
        ylabel="Ratio of transactions",
        xscale="log",
        yscale="log",
        title=(
            "Distribution of the amount of "
            f"{"fraudulent" if class_id else "legitimate"} transactions"
        ),
    )

# %% [markdown]
#
# We cannot conclude a particular pattern in the distribution of the amount of the
# transactions apart from the fact that the fraudulent transactions tend to not have
# really large amounts. This information could be useful: if we train a predictive
# model on these data, we should consider that we do not know how our predictive model
# will behave on fraudulent transactions with large amounts in the future. It might be
# worth considering to have a specific treatment for those transactions.
#
# ## Addressing the problem with a business metric
#
# Now, we create the business metric that depends on the amount of each transaction.
#
# The gain of a legitimate transaction is quite easy to define since it is a commission
# that the bank receives. Here, we define it to be 2% of the amount of the transaction.
# Similarly, there is no gain at refusing a fraudulent transaction: the bank does not
# receive money from external actors or clients in this case.
#
# Defining a cost for refusing a legitimate transaction or accepting a fraudulent
# transaction is more complex. If we accept a fraudulent transaction, the bank loses the
# amount of the transaction. There is also an extract cost involved that is an
# aggregation of several other costs: the cost of the fraud investigation, the cost of
# the customer support, and the cost related to brand reputation damage. Those
# additional should be defined by the data scientist in collaboration with the business
# stakeholders. A similar approach should be taken for the cost of refusing a
# legitimate: the cost of the customer support, the cost of the customer
# dissatisfaction, and the cost of the customer churn.

# %%
# Commission received for each accepted legitimate transaction
commission_transaction_gain = 0.02
# Average cost of accepting a fraudulent transaction
avg_accept_fraud_cost = 20
# Average cost of refusing a legitimate transaction
avg_refuse_legit_cost = 10


def business_gain_func(y_true, y_pred, amount):
    """Business metric to optimize.

    The amount computed in this function are expressed in terms of gain. It means that
    the diagonal entry of the cost matrix C_00 and C_11 are positive values and the
    other entries are negative values. The off-diagonal entries are however negative
    values.
    """
    true_negative_c00 = (y_pred == 0) & (y_true == 0)
    false_negative_c01 = (y_pred == 0) & (y_true == 1)
    false_positive_c10 = (y_pred == 1) & (y_true == 0)
    true_positive_c11 = (y_pred == 1) & (y_true == 1)

    accept_legitimate = (amount[true_negative_c00] * commission_transaction_gain).sum()
    accept_fraudulent = -(
        amount[false_negative_c01].sum()
        + (false_negative_c01 * avg_accept_fraud_cost).sum()
    )
    refuse_legitimate = -(false_positive_c10 * avg_refuse_legit_cost).sum()
    refuse_fraudulent = (true_positive_c11 * 0).sum()

    return accept_legitimate + accept_fraudulent + refuse_legitimate + refuse_fraudulent


# %% [markdown]
#
# From this business metric, we create a scikit-learn scorer that given a fitted
# classifier and a test set compute the business metric. This scorer is handy because it
# can be used in meta-estimators, grid-search, and cross-validation.
#
# To create this scorer, we use the :func:`~sklearn.metrics.make_scorer` factory. The
# metric defined above request the amount of each transaction. This variable is an
# additional metadata to be passed to the scorer and we need to use metadata routing to
# take into account this information.

# %%
import sklearn
from sklearn.metrics import make_scorer

sklearn.set_config(enable_metadata_routing=True)
business_gain_scorer = make_scorer(business_gain_func).set_score_request(amount=True)

# %% [markdown]
#
# So at this stage, we see that the amount of the transaction is used twice: once as a
# feature to train our predictive model and once as a metadata to compute the the
# business metric and thus the statistical performance of our model. When used as a
# feature, we are only required to have a column in `data` that contains the amount of
# each transaction. To use this information as metadata, we need to have an external
# variable that we can pass to the scorer or the model that internally routes this
# metadata to the scorer. So let's create this variable.

# %%
amount = credit_card["Amount"].to_numpy()

# %% [markdown]
#
# ## Investigate baseline policies
#
# Before to train a machine learning model, we investigate some baseline policies to
# serve as reference. Also, we prepare our dataset, to have a left-out test set to
# evaluate the performance of our predictive model.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test, amount_train, amount_test = (
    train_test_split(
        data, target, amount, stratify=target, test_size=0.5, random_state=42
    )
)

# %% [markdown]
#
# The first baseline policy to evaluate is to check the performance of a policy that
# always accepts the transaction. We recall that class "0" is the legitimate class and
# class "1" is the fraudulent class.

# %%
from sklearn.dummy import DummyClassifier

always_accept_policy = DummyClassifier(strategy="constant", constant=0)
always_accept_policy.fit(data_train, target_train)
benefit = business_gain_scorer(
    always_accept_policy, data_test, target_test, amount=amount_test
)
print(f"Benefit of the 'always accept' policy: {benefit:,.2f}€")

# %% [markdown]
#
# A policy that considers all transactions as legitimate would create a profit of
# around 216,000€. We make the same evaluation for a classifier that predicts all
# transactions as fraudulent.

# %%
always_reject_policy = DummyClassifier(strategy="constant", constant=1)
always_reject_policy.fit(data_train, target_train)
benefit = business_gain_scorer(
    always_reject_policy, data_test, target_test, amount=amount_test
)
print(f"Benefit of the 'always reject' policy: {benefit:,.2f}€")

# %% [markdown]
#
# Such a policy would entail a catastrophic loss: around 1,421,000€. This is
# expected since the vast majority of the transactions are legitimate and the
# policy would refuse them at a non-trivial cost.
#
# Now, we evaluate the oracle model that would not make any mistake at all.

# %%
business_score = business_gain_func(
    target_test,
    target_test,
    amount=amount_test,
)
print(f"Benefit of oracle decisions (not reachable):  {business_score:,.2f}€")


# %% [markdown]
#
# This perfect model would make a profit of around 251,000€.
#
# Therefore, we conclude that a predictive model that a model which adapts the
# accept/reject decisions on a per transaction basis should ideally allow us to make a
# profit larger than the ~216,000€ and will be capped by an amount of ~251,000€ of the
# best of our constant baseline policies.

# %% [markdown]
#
# ## Training predictive models
#
# ### Tuned logistic regression on a proper scoring rule
#
# We start training a logistic regression model with the default decision threshold at
# 0.5. Here we tune the hyperparameter `C` of the logistic regression with a proper
# scoring rule (the log loss) to ensure that the model's probabilistic predictions
# returned by its `predict_proba` method are as accurate as possible, irrespectively of
# the choice of the value of the decision threshold.

# %%
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
    f"{business_gain_scorer(model, data_test, target_test, amount=amount_test):,.2f}€"
)

# %% [markdown]
#
# The business metric shows that our predictive model with a default decision threshold
# is already winning over the baseline in terms of profit and it would be already
# beneficial to use it to accept or reject transactions instead of accepting all
# transactions.
#
# ### Tuned logistic regression with optimal decision threshold
#
# From the research paper from Charles Elkan [1]_, we know that the optimal decision
# threshold can be computed given the following two assumptions:
#
# - the probabilistic classifier is well-calibrated,
# - the business metric can be expressed as a cost matrix.
#
# When defining our business metric, we have already expressed it as a cost matrix. So
# to use the approach described in [1]_, we only need to check the calibration of our
# model. In the previous section, we already tuned the hyperparameter of the logistic
# regression using a proper scoring rule that should help towards getting a
# well-calibrated model. As a first step, we assume that our classifier is
# well-calibrated. Later, we will add an extra calibration step to check if it improves
# the performance of our model in terms of the business metric.
#
# The optimal decision threshold proposed by Charles Elkan in [1]_ is defined as
# follows:

# %%


def elkan_optimal_threshold(amount):
    """Compute the Elkan-optimal threshold for a given amount.

    The values are expressed as costs. Therefore, the diagonal entries of the cost
    matrix (C_00 and C_11) are negative values and the off-diagonal values are positive.
    """
    c00 = -commission_transaction_gain * amount  # Accepting a legitimate transaction
    c01 = amount + avg_accept_fraud_cost  # Accepting a fraudulent transaction
    c10 = avg_refuse_legit_cost  # Refusing a legitimate transaction
    c11 = 0  # Refusing a fraudulent transaction
    optimal_threshold = (c10 - c00) / (c10 - c00 + c01 - c11)
    return optimal_threshold


# %% [markdown]
#
# Therefore, given a transaction amount, an optimal threshold can be computed for each
# transaction amount. Let's plot the distribution of the optimal threshold for the
# transactions in the test set. In addition, we plot the optimal threshold as a function
# of the transaction amount.

# %%
_, ax = plt.subplots(ncols=2, figsize=(14, 6))

ax[0].hist(elkan_optimal_threshold(amount_train), bins=50, edgecolor="black")
ax[0].set(
    xlabel="Optimal threshold",
    ylabel="Number of transactions",
    title="Optimal threshold distribution",
)

x = np.linspace(amount_train.min(), amount_train.max(), 1_000)
ax[1].plot(x, elkan_optimal_threshold(x))
_ = ax[1].set(
    xlabel="Amount (€)",
    ylabel="Optimal threshold",
    title="Optimal threshold in function of the transaction amount",
)

# %% [markdown]
#
# We see that the optimal threshold varies from ~0.02 to ~0.33 depending on the amount
# of the transaction. Looking at the optimal threshold as a function of the transaction
# amount, we see that the optimal threshold decreases as the amount of the transaction
# increases. Therefore, it means that we declare a transaction as fraudulent with a
# lower probability when the amount of the transaction is higher.
#
# As a first experiment, we set the decision threshold to the mean of the optimal
# threshold for the transactions in the test set. Let's check the value of this
# threshold.

# %%
elkan_optimal_threshold(amount_train).mean()

# %% [markdown]
#
# Let's use this global threshold to change the decision threshold of our logistic
# regression model and evaluate the performance of our model in terms of the business
# metric.

# %%
from sklearn.model_selection import FixedThresholdClassifier

fixed_elkan_model = FixedThresholdClassifier(
    estimator=model.best_estimator_,
    threshold=elkan_optimal_threshold(amount_train).mean(),
).fit(data_train, target_train)

business_score = business_gain_scorer(
    fixed_elkan_model, data_test, target_test, amount=amount_test
)

print(
    f"Benefit of logistic regression with a fixed mean theoretical threshold: "
    f"{business_score:,.2f}€"
)

# %% [markdown]
#
# We see that using a more optimal threshold than the default one increases the profit
# of our model. Since, we made the assumption that our model is well-calibrated, we
# could now check that it was really the case.

# %%
from sklearn.calibration import CalibrationDisplay

disp = CalibrationDisplay.from_estimator(
    model.best_estimator_,
    data_test,
    target_test,
    strategy="quantile",
    n_bins=5,
    name="Tuned logistic regression",
)
_ = disp.ax_.set(xlim=(1e-7, 0.03), ylim=(1e-7, 0.03), xscale="log", yscale="log")

# %% [markdown]
#
# Our model is not perfectly calibrated. Before to tune the decision threshold, we
# might try to improve the model calibration. However, it should be noted that since
# we have little data in the fraudulent class, the classifier might be difficult to
# calibrate since we need to make an internal cross-validation.

# %%
from sklearn.calibration import CalibratedClassifierCV

calibrated_estimator = CalibratedClassifierCV(
    model.best_estimator_, method="isotonic"
).fit(data_train, target_train)
disp = CalibrationDisplay.from_estimator(
    calibrated_estimator, data_test, target_test, strategy="quantile", n_bins=5
)
_ = disp.ax_.set(xlim=(1e-7, 0.03), ylim=(1e-7, 0.03), xscale="log", yscale="log")

# %% [markdown]
#
# Now that our model has been through a calibration step, we can check the performance
# of our model with the optimal threshold.

# %%
fixed_elkan_model = FixedThresholdClassifier(
    estimator=calibrated_estimator,
    threshold=elkan_optimal_threshold(amount_train).mean(),
).fit(data_train, target_train)

business_score = business_gain_scorer(
    fixed_elkan_model, data_test, target_test, amount=amount_test
)

print(
    f"Benefit of recalibrated logistic regression with a fixed mean theoretical "
    f" threshold:  {business_score:,.2f}€"
)

# %% [markdown]
#
# It seems that this extra calibration step did improve the performance of our model in
# terms of the business metric.
#
# ### Tune decision threshold by business metric optimization
#
# In the previous section, we presented a method to compute the optimal decision
# threshold but it relied on the assumption that the probabilistic classifier is
# well-calibrated and that the business metric can be expressed as a cost matrix.
#
# If those assumptions are not met, we can instead tune the decision threshold by
# directly optimizing the business metric. This optimization is done through a
# grid-search over the decision threshold involving a cross-validation. The class
# :class:`~sklearn.model_selection.TunedThresholdClassifierCV` is in charge of
# performing this optimization.

# %%
from sklearn.model_selection import TunedThresholdClassifierCV

tuned_model = TunedThresholdClassifierCV(
    estimator=model.best_estimator_,
    scoring=business_gain_scorer,
    thresholds=100,
    n_jobs=2,
)
tuned_model

# %% [markdown]
#
# Since our business scorer requires the amount of each transaction, we need to pass
# this information in the `fit` method. The
# :class:`~sklearn.model_selection.TunedThresholdClassifierCV` is in charge of
# automatically dispatching this metadata to the underlying scorer.

# %%
tuned_model.fit(data_train, target_train, amount=amount_train)

# %% [markdown]
#
# Let's compare the decision threshold found by the model compared to our fixed global
# threshold from the previous section.

# %%
tuned_model.best_threshold_

# %% [markdown]
#
# Now, let's check the performance of our model with the tuned decision threshold on
# the test set and using the business metric.

# %%
print(
    "Benefit of logistic regression with a tuned threshold: "
    f"{business_gain_scorer(tuned_model, data_test, target_test, amount=amount_test):,.2f}€"
)

# %% [markdown]
#
# We see that the obtained profit is slightly higher than the one obtained with the
# fixed global threshold.
#
# ### Variable optimal threshold
#
# As we previously mentioned, theoretically, there is an optimal threshold for each
# transaction amount. Let's write a similar class than the `FixedThresholdClassifier`
# but that uses different thresholds depending on the amount of the transaction.

# %%


class VariableThresholdClassifier:

    def __init__(self, classifier, variable_threshold):
        self.classifier = classifier
        self.variable_threshold = variable_threshold

    def fit(self, X, y):
        return self

    def predict(self, X, amount):
        proba = self.classifier.predict_proba(X)[:, 1]
        return (proba >= self.variable_threshold(amount)).astype(int)


# %% [markdown]
#
# This estimator takes a trained classifier and call the optimal threshold function for
# each of the predictions and compare it with the classifier's probability predictions.
# We now evaluate the performance of this model on the test set.

# %%
business_score = business_gain_func(
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

# %% [markdown]
#
# We see that the profit is almost the same as the one obtained with the tuned threshold
# model. However, we did not take care about the calibration of the underlying model
# while as for the fixed threshold model, this is an assumption. So let's used the
# recalibrated model to see if it improves the performance of our model.

# %%
business_score = business_gain_func(
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

# %% [markdown]
#
# We see that the recalibrated model improve the performance of our model in terms of
# the business metric.
#
# ## Conclusion
# TODO: Add a conclusion
#
# ## References
#
# .. [1] `Charles Elkan, "The Foundations of Cost-Sensitive Learning",
#        International joint conference on artificial intelligence.
#        Vol. 17. No. 1. Lawrence Erlbaum Associates Ltd, 2001.
#        <https://cseweb.ucsd.edu/~elkan/rescale.pdf>`_

# %%
