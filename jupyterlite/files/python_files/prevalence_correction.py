# %% [markdown]
#
# # Handling prevalence shift in imbalanced classification problems
#
# The purpose of this study is to illustrate how to handle extreme class
# imbalance in the common setting where the **data acquisition process itself
# corrupts the prevalence of the rare class**.
#
# This setting is quite common in practice: for instance, to develop a
# computer-aided diagnosis system, the data acquisition process might collect
# data (feature values) for all known cases of a given rare disease of interest
# (the positive class): **since the positive cases are rare, we don't want to
# waste any of them** and want include them all in our dataset. However, it
# would be **impossible to collect feature values for all the negative
# subjects** in the target population because they are too many of them: it
# would be too costly and might also cause ethical and privacy issues.
#
# Instead, we sample negative subjects from the population at random to measure
# their feature values only for those (and check that they do not have the
# disease).
#
# As a result, **the prevalence of the positive class in the collected dataset
# does not reflect the prevalence of the positive class in the target
# population**. Still, we do want to train a probabilistic model on the finite
# collected data that achieves the best performance possible when deployed on
# the target population. Furthermore, we want to conduct this performance
# evaluation despite the fact that we cannot directly measure the performance
# on the full target population at once. We will therefore need to:
#
# - **adapt the training procedure** to take the prevalence shift into account
#   to learn a model that yields meaningful probabilistic predictions on the
#   target population once deployed;
# - **adapt the evaluation procedure** to ensure that our performance
#   evaluation computed from a test subset of our observed data points can
#   accurately reflect the expected performance of our model once deployed
#   despite the prevalence shift.
#
# The lack of match of prevalence between the collected dataset and the
# deployment setting is a result of cost or computational constraints that
# prevent us from training and evaluating our predictive model directly on the
# full target population. It is not the result of a modeling choice of the
# data-scientist.
#
# Note that the classes are extremely imbalanced in the target population but
# can also be (slightly) imbalanced in the collected dataset. We do not need
# the collected dataset to be perfectly balanced.
#
# In this study, we will illustrate how to handle such a setting by correcting
# the probabilistic classifier estimated from the observed data to be aligned
# with the target population. We will use synthetic data generated from a known
# data generating process so as to make it possible to check that our proposed
# training and evaluation methods can achieve that objective.

# %% [markdown]
# ## Data generating process
#
# Let's define a "true" data generating process that represent some fundamental
# mechanism about the world. The true data generating process is generally
# unknown, and the goal of machine learning is to approximate it as closely as
# possible from a finite sample of data points.
#
# Here, we want to simulate a binary classification problem where the target
# variable is a binary variable with a positive class that is very rare.
#
# The typical application domain could be medical diagnosis where the feature
# values are physiological measurements and the target variable is a binary
# indicator for the presence of a rare disease of interest. Other application
# domains with extreme imbalance are fraud detection, credit scoring,
# predictive maintenance, etc.
#
# For the sake of simplicity, we start this study by assuming that the data
# generating process is a linear model with a logistic link function: the
# features influence the probability of developing the disease but we expect
# them to provide only a partial information about the true risk as other
# unobserved factors may also influence disease development. We assume the
# unobserved factors to be independent of the observed features and all
# distribution to be stationary over time.
#
# We will later relax the linearity assumption but other assumptions will still
# apply.

# %%
import numpy as np
import pandas as pd
from scipy.special import expit, logit

rng = np.random.default_rng(0)
dtype = np.float32  # use float32 to save memory
n_features = 5
true_coef = rng.normal(size=n_features).astype(dtype)
true_intercept = -6  # This will make the positive class very rare.


def sample_from_linear_model(true_coef, true_intercept, n_samples, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(dtype)
    z = X @ true_coef
    true_positive_proba = expit(z + true_intercept)
    y = rng.binomial(n=1, p=true_positive_proba)
    true_proba = np.hstack(
        [1 - true_positive_proba[:, np.newaxis], true_positive_proba[:, np.newaxis]]
    )

    # create pandas data structures for convenience
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y, name="target")
    return X, y, true_proba


# %% [markdown]
#
# The world can generate a lot of data from a fixed stationary process. Some of that data
# cannot be accessed at the time of the study but we generate it anyway to be able to
# compute metrics on the future population.

# %%
n_samples = 3_000_000
X_past, y_past, true_proba_past = sample_from_linear_model(
    true_coef, true_intercept, n_samples, seed=0
)
X_future, y_future, true_proba_future = sample_from_linear_model(
    true_coef, true_intercept, n_samples, seed=1
)

# %%
X_past.memory_usage().sum() / 1e6  # in MB

# %%
y_past.sum()

# %%
true_positive_rate_past = y_past.mean()
true_positive_rate_past

# %% [markdown]
#
# Let us start this study by cheating and compute the metrics obtained from the
# true probabilities computed from the data generating process for the future
# feature values.

# %%
true_positive_rate_future = y_future.mean()
true_positive_rate_future

# %%
pd.Series(true_proba_future[:, 1]).describe()

# %%
from sklearn.metrics import roc_auc_score, log_loss

roc_auc_score(y_future, true_proba_future[:, 1])

# %%
log_loss(y_future, true_proba_future)

# %%
log_loss(y_past, true_proba_past)

# %% [markdown]
#
# ### Questions:
# - What is the best possible value for the ROC AUC score (in general)?
# - What is the best possible value for the log-loss (in general)?
# - Why cannot we reach these values even when using the true probabilities?
# - What is the name name of the error rate obtained when computed from the
#   true probabilities?

# %%
# TODO: write your answers here before scrolling down.
#
#
#
#
#
#
#
#
#
#
#
#

# %% [markdown]
#
# ### Answers:
# - The best possible value for the ROC AUC score is 1.0.
# - The best possible value for the log-loss is 0.0.
# - We cannot reach these values because the true probabilities reflect some
#   inherent uncertainty in the data: the value of the target variable is not
#   deterministic given the available features. The outcome can also be
#   influenced by unobserved factors that are not captured by the features.
# - The error rate obtained from the true probabilities is called the "Bayes
#   error rate" or "irreducible error rate".


# %% [markdown]
#
# Because we are curious, let's continue with the cheating and check that a fit
# of logistic regression on the full population (with nearly unlimited data
# points) would be able to approximately recover the true coefficients and
# intercept.

# %%
from sklearn.linear_model import LogisticRegression

cheating_model = LogisticRegression(penalty=None).fit(X_future, y_future)


# %%
class ModelComparator:
    def __init__(self, X, y, context_name, sample_weight=None):
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.context_name = context_name
        self.models = {}
        self.evaluation_records = []

    def score_model(self, model_name, predicted_proba):
        self.evaluation_records.append(
            {
                "Model": model_name,
                f"ROC AUC ({self.context_name})": roc_auc_score(
                    self.y,
                    predicted_proba[:, 1],
                    sample_weight=self.sample_weight,
                ),
                f"log-loss ({self.context_name})": log_loss(
                    self.y,
                    predicted_proba,
                    sample_weight=self.sample_weight,
                ),
            }
        )
        return self

    def register_linear_data_generating_model(self, true_coef, true_intercept):
        self.true_coef = true_coef
        self.true_intercept = true_intercept
        oracle_proba = expit(self.true_intercept + self.X @ self.true_coef).values
        oracle_proba = np.hstack(
            [1 - oracle_proba[:, np.newaxis], oracle_proba[:, np.newaxis]]
        )
        self.score_model("Data generating model", oracle_proba)
        return self

    def register_model(self, model_name, model):
        self.models[model_name] = model
        self.score_model(model_name, model.predict_proba(self.X))
        return self

    def score_table(self):
        return pd.DataFrame(self.evaluation_records).round(6).set_index("Model")

    def plot_linear_model_parameters(self):
        column_data = {}
        if hasattr(self, "true_coef"):
            column_data["Data generating model"] = np.hstack(
                (self.true_intercept, self.true_coef)
            )
        column_data.update(
            {
                model_name: np.hstack((model.intercept_, model.coef_.flatten()))
                for model_name, model in self.models.items()
            }
        )
        pd.DataFrame(
            column_data,
            index=np.hstack(["intercept", self.X.columns]),
        ).plot.barh().set(
            title="Comparison of the true and learned model parameters",
            xlabel="Parameter value",
            ylabel="Parameter name",
        )


population_comparator = ModelComparator(X_future, y_future, context_name="population")
population_comparator.register_linear_data_generating_model(true_coef, true_intercept)
population_comparator.register_model("LogReg fit on future data", cheating_model)
population_comparator.plot_linear_model_parameters()

# %% [markdown]
#
# We can see that the learned model parameters nearly recover the data
# generating parameters. This is expected since the logistic regression model
# without penalty is well specified for this data generating process and that
# we have a large number of data points.

# %% [markdown]
#
# ## Prevalence shift induced by the data acquisition process
#
# The following code simulate what could happen in practice when working in
# such a setting: we subsample the simulated population to collect all the
# positive cases and a random sample of negatives from the past data. Mere
# mortal data scientists cannot `pd.read_csv` from the future, unfortunately.
#
# We subsample only the negative cases (control group) to reflect the fact that
# the negative data is less interesting than the rare positive cases: as such
# the negative data is in general not archived fully in databases, or even, the
# features of most of the negative cases are not acquired at all in the first
# place.

# %%
mask_positive_target = y_past == 1
X_positive = X_past[mask_positive_target].copy()
y_positive = y_past[mask_positive_target].copy()

rng = np.random.default_rng(0)
negative_indices = rng.choice(
    np.flatnonzero(~mask_positive_target),
    size=3 * len(y_positive),
    replace=False,
)

X_observed = pd.concat([X_positive, X_past.iloc[negative_indices]])
y_observed = pd.concat([y_positive, y_past.iloc[negative_indices]])

X_observed.shape
# %%
observed_positive_rate = y_observed.sum() / len(y_observed)
observed_positive_rate

# %% [markdown]
#
# Now that the data is collected, we can train-test split it to reflect what a
# data scientist would do in practice to train and evaluate their model from
# the observed data.

# %%
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X_observed, y_observed, test_size=0.5, random_state=0
)

# %%
logreg_params = dict(penalty=None, tol=1e-8)
logreg_uncorrected = LogisticRegression(**logreg_params).fit(X_train, y_train)

population_comparator.register_model("Uncorrected LogReg", logreg_uncorrected)
population_comparator.plot_linear_model_parameters()
population_comparator.score_table()


# %%
class_weight_for_prevalence_correction = {
    0: (1 - true_positive_rate_past) / (1 - y_train.mean()),
    1: true_positive_rate_past / y_train.mean(),
}
class_weight_for_prevalence_correction

# %%
logreg_weighted = LogisticRegression(
    class_weight=class_weight_for_prevalence_correction, **logreg_params
).fit(X_train, y_train)

population_comparator.register_model("Weight-corrected LogReg", logreg_weighted)
population_comparator.plot_linear_model_parameters()
population_comparator.score_table()

# %% [markdown]
#
# Let's check that we can get exactly the same results using `sample_weight` in
# `fit` instead of `class_weight` in the constructor.

# %%
sample_weight_for_prevalence_correction = np.where(
    y_train == 0,
    class_weight_for_prevalence_correction[0],
    class_weight_for_prevalence_correction[1],
)
logreg_weighted2 = LogisticRegression(**logreg_params).fit(
    X_train, y_train, sample_weight=sample_weight_for_prevalence_correction
)
logreg_weighted2.coef_, logreg_weighted2.intercept_

# %%
np.allclose(logreg_weighted.coef_, logreg_weighted2.coef_)

# %%
np.allclose(logreg_weighted.intercept_, logreg_weighted2.intercept_)

# %% [markdown]
#
# From the above results, it seems that the uncorrected linear model and the
# weight-corrected linear models only significantly differ by the value of the
# intercept parameter.
#
# Indeed, it can be shown that the intercept of a logistic regression model can
# be corrected by shifting it by the difference of the logits of the prevalence
# in the target population and the training set.
#
# See for instance: https://stats.stackexchange.com/a/68726/2150
#
# Recall that the `logit` function is defined as:
#
# $$ \text{logit}(p) = \log\left(\frac{p}{1 - p}\right) \in \mathbb{R} \text{
# for any } p \in [0, 1] $$
#
# Let's implement this intercept correction as follows:

# %%
from scipy.special import logit


logreg_intercept_corrected = LogisticRegression(**logreg_params).fit(X_train, y_train)

intercept_shift = logit(true_positive_rate_past) - logit(y_train.mean())
logreg_intercept_corrected.intercept_ += intercept_shift

population_comparator.register_model(
    "Intercept-corrected LogReg", logreg_intercept_corrected
)
population_comparator.plot_linear_model_parameters()
population_comparator.score_table()

# %% [markdown]
#
# We recover an intercept value that is very close to the intercept of the
# data generating process.

# %% [markdown]
#
# Let's now consider a more generic post-hoc prevalence correction that does
# not require the base model to be a logistic regression model with an explicit
# `intercept_` parameter.
#
# Charles Elkan proposed a closed-form formula to correct the predicted
# probabilities of any binary classifier:


# %%
def elkan_prevalence_correction(
    uncorrected_positive_proba, target_prevalence, observed_prevalence
):
    """Post-hoc prevalence correction for binary classifiers.

    Given a classifier to be trained on a class-conditional subsampled dataset
    and an estimate of the prevalence of the positive class in the target
    population, estimate the conditional probability of the positive class for
    each data point in the test set.

    This meta-estimator implements the formula of Theorem 2 of [Elkan 2001]:

    The Foundations of Cost-Sensitive Learning, Charles Elkan, IJCAI 2001
    https://cseweb.ucsd.edu/~elkan/rescale.pdf

    p' = b'(p - pb) / (b - pb + b'p - b'b)

    with:

    - p' is the corrected estimate of P_target(y=1|X=x) on the target population.
    - p is the observed estimate of P_data(y=1|X=x) on the training set.
    - b' is the prevalence of the positive class in the target population.
    - b is the observed prevalence of the positive class measured in the
      training set.
    """
    # b'(p - pb)
    numerator = target_prevalence * (
        uncorrected_positive_proba - (uncorrected_positive_proba * observed_prevalence)
    )
    # b - pb + b'p - b'b
    denominator = (
        observed_prevalence
        - (uncorrected_positive_proba * observed_prevalence)
        + (target_prevalence * uncorrected_positive_proba)
        - (target_prevalence * observed_prevalence)
    )
    return numerator / denominator


# %% [markdown]
#
# We can wrap this correction function into a meta-estimator compatible
# with the scikit-learn API:

# %%
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class PostHocPrevalenceCorrection(ClassifierMixin, BaseEstimator):

    def __init__(self, estimator=None, target_positive_rate=0.5):
        self.estimator = estimator
        self.target_positive_rate = target_positive_rate

    def fit(self, X, y, **fit_params):
        if self.estimator is None:
            estimator = LogisticRegression()
        else:
            estimator = clone(self.estimator)

        self.estimator_ = estimator.fit(X, y, **fit_params)
        self.observed_positive_rate_ = y.mean()
        return self

    def predict_proba(self, X):
        uncorrected_proba = self.estimator_.predict_proba(X)
        corrected_positive_proba = elkan_prevalence_correction(
            uncorrected_proba[:, 1],
            self.target_positive_rate,
            self.observed_positive_rate_,
        )
        corrected_proba = np.zeros_like(uncorrected_proba)
        corrected_proba[:, 1] = corrected_positive_proba
        corrected_proba[:, 0] = 1 - corrected_positive_proba
        return corrected_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int32)


logreg_post_hoc = PostHocPrevalenceCorrection(
    estimator=LogisticRegression(**logreg_params),
    target_positive_rate=true_positive_rate_past,
).fit(X_train, y_train)

# %% [markdown]
#
# For a Logistic Regression model, this generic post-hoc imbalance correction
# should be strictly equivalent to our previously introduced post-hoc
# correction of the intercept parameter:

# %%
np.allclose(
    logreg_post_hoc.predict_proba(X_future),
    logreg_intercept_corrected.predict_proba(X_future),
)

# %% [markdown]
#
# A proof of the mathematical equivalence is given as an appendix at the end of
# this notebook.

# %% [markdown]
#
# Therefore we should get exactly the same evaluation metric values as before:

population_comparator.register_model("Post-hoc corrected LogReg", logreg_post_hoc)
population_comparator.score_table()

# %% [markdown]
#
# ### Questions:
#
# Consider the two kinds of post-hoc prevalence correction presented above:
# - what happens when we pass `target_positive_rate=y_train.mean()`?
# - is it possible to get `predict_proba` values that are not in $[0, 1]$ by
#   setting extreme values for `target_positive_rate`?
# - why is the ROC-AUC score not affected by any of the post-hoc correction methods?
#
# Recall that the `expit` function is takes any real number as input and
# returns a value in [0, 1]:
#
# $$ \text{expit}(x) = \frac{1}{1 + e^{-x}} \in [0, 1] \text{ for any } x \in
# \mathbb{R} $$
#
# The `expit` function inverse function is the logit function.

# %%
#
#
#
#
#
#
#
# TODO: do not read the solution before writing done your answers ;)

# %% [markdown]
#
# ### Answers:
#
# - When we pass `target_positive_rate=y_train.mean()`, the intercept
#   correction `logit(target_positive_rate) - y_train.mean()` will be zero,
#   hence no correction is applied.
# - Similarly, setting $b = b'$ in the formula of the docstring of
#   `elkan_prevalence_correction` will cause the expression to simplify to $p =
#   p'$.
# - Shifting the intercept of a logistic regression model can never make its
#   predictions go outside of the [0, 1] interval since is prediction function
#   is `expit(X @ coef + intercept)`: the `expit` function is defined for all
#   real numbers and maps them to the [0, 1] interval.
# - For `elkan_prevalence_correction`, the result is less obvious but we can
#   check empirically that the corrected probabilities are also in [0, 1] by
#   tweaking the inputs of the plotting snippet below.
# - Both correction method are monotonic transformations of the predicted
#   probabilities, hence they preserve the order of the predictions. As a result,
#   the relative ranking of the instances remains unchanged and therefore the ROC-AUC
#   score is not affected by any of the post-hoc correction methods.

# %%
import matplotlib.pyplot as plt

p = np.linspace(0, 1, 100)
for target_prevalence, observed_prevalence in [
    (0.01, 0.25),
    (0.1, 0.25),
    (0.01, 0.01),
    (0.25, 0.01),
]:
    plt.plot(
        p,
        elkan_prevalence_correction(
            p,
            target_prevalence=target_prevalence,
            observed_prevalence=observed_prevalence,
        ),
        label=(
            f"Target prevalence: {target_prevalence:.2f}, "
            f"observed prevalence: {observed_prevalence:.2f}"
        ),
    )
plt.xlabel("Observed P(y=1|X=x)")
plt.ylabel("Corrected P(y=1|X=x)")
plt.title("Elkan Prevalence Correction")
_ = plt.legend()

# %% [markdown]
#
# ## Evaluating models on observed test data
#
# As we saw above, we have several ways to correct the predictions of a
# classifier to account for difference of class distribution between the
# observed training data and the target population.
#
# However, data-scientists do not have access to the future population to
# assess the expected performance of their model on the future population.
#
# Instead, they can only evaluate the model on the observed test data. However,
# the observed test data: in our case, the number of negatives cases is much
# lower in the observed data (train or test) than in the target population.
#
# If we naively evaluate the model on the observed test data, we will get
# misleading results: the model will be evaluated on a test set that does not
# reflect the true class prevalence, hence calibration sensitive losses such as
# the log-loss will give very inaccurate estimates of the population log-loss:

log_loss(y_test, logreg_uncorrected.predict_proba(X_test))

# %%
log_loss(y_future, logreg_uncorrected.predict_proba(X_future))

# %% [markdown]
#
# Therefore, we need to apply the same class ratio correction to the evaluation
# set. Scikit-learn metric functions usually do not provide a `class_weight`
# parameter but they often provide a `sample_weight` parameter.
#
# Let's evaluate the log-loss of the uncorrected model on the test set with the
# sample weights and check that it correctly estimate the population log-loss
# that same model:


# %%
sample_weight_test = np.where(
    y_test == 0,
    (1 - true_positive_rate_past) / (1 - y_test.mean()),
    true_positive_rate_past / y_test.mean(),
)

# %%
log_loss(
    y_test, logreg_uncorrected.predict_proba(X_test), sample_weight=sample_weight_test
)

# %%
log_loss(y_future, logreg_uncorrected.predict_proba(X_future))

# %% [markdown]
#
# The test data is a comparatively small finite set, so the estimation of the
# population log-loss is not perfect but close enough and the weighting definitely
# helps obtain metric values that are aligned with the expected population log-loss.
#
# Let's now evaluate one of the corrected models:

# %%
log_loss(
    y_test,
    logreg_intercept_corrected.predict_proba(X_test),
    sample_weight=sample_weight_test,
)

# %%
log_loss(y_future, logreg_intercept_corrected.predict_proba(X_future))

# %% [markdown]
#
# We can see that weighting the test observed makes it possible to approximate
# the population log-loss (at least up to 2 to 3 decimal places in this case).
#
# Let's improve our evaluation tools to consolidate all scores for all models:

# %%
# TODO


# %% [markdown]
#
# ## Fitting non-linear models

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

gbdt_uncorrected = HistGradientBoostingClassifier(random_state=0).fit(X_train, y_train)
population_comparator.register_model("Uncorrected GBDT", gbdt_uncorrected)
population_comparator.score_table()

# %%
gbdt_weighted = HistGradientBoostingClassifier(
    random_state=0,
    class_weight=class_weight_for_prevalence_correction,
).fit(X_train, y_train)
# %%
population_comparator.register_model("Weight-corrected GBDT", gbdt_weighted)
population_comparator.score_table()

# %%
gbdt_post_hoc = PostHocPrevalenceCorrection(
    estimator=HistGradientBoostingClassifier(random_state=0),
    target_positive_rate=true_positive_rate_past,
).fit(X_train, y_train)

# %%
population_comparator.register_model("Post-hoc corrected GBDT", gbdt_post_hoc)
population_comparator.score_table()

# %% [markdown]
#
# ## Take away messages
#
# - It is possible to correct a binary classifier trained on observed data to
#   correctly account for differences of prevalence between the training set
#   and the target populations.
# - This correction can be achieved either via training the model with
#   appropriate class or sample weights or by applying a post-hoc correction
#   method to the predicted probabilities.
# - In the case of a linear model, the post-hoc correction can be achieved by
#   adjusting the model's intercept based on the difference of the logits of
#   the two prevalence values.
# - For other estimators that do not have an explicit intercept parameter, this
#   can be achieved by applying a (monotonic) transformation to the predicted
#   probabilities.
# - Since this correction is monotonic, the order of the predicted
#   probabilities is preserved and therefore the ROC AUC score is not affected.
# - It is possible to estimate the expected performance of the model on the
#   target population only from the finite, prevalence-shifted sample by
#   applying the same weight-based correction to the evaluation metrics.
#
# Open question: the weight-based training prevalence correction and the
# post-training closed-form prevalence correction methods can yield slightly
# different results in practice (although both should converge to the Bayes
# optimal classifier as the sample size increases). Are there reasons to favor
# one over the other in the finite sample case?

# %% [markdown]
#
# ## Appendix: equivalence between the two post-hoc correction methods
#
# In the following, we show that the logit based shift of the intercept is
# equivalent to the Elkan's formula.
#
# Let's first check empirically with matplotlib:


def logit_prevalence_correction(p, target_prevalence, observed_prevalence):
    corrected_logits = logit(p) + logit(target_prevalence) - logit(observed_prevalence)
    return expit(corrected_logits)


p = np.linspace(0, 1, 100)

target_prevalence = 0.01
observed_prevalence = 0.25
plt.plot(
    p,
    elkan_prevalence_correction(
        p,
        target_prevalence=target_prevalence,
        observed_prevalence=observed_prevalence,
    ),
    label="Elkan's formula",
)
plt.plot(
    p,
    logit_prevalence_correction(
        p,
        target_prevalence=target_prevalence,
        observed_prevalence=observed_prevalence,
    ),
    linestyle="--",
    label="Difference of prevalence logits",
)
plt.xlabel("Observed P(y=1|X)")
plt.ylabel("Corrected P(y=1|X)")
_ = plt.legend()


# %% [markdown]
#
# Let's ask Claude to check that $p' = \frac{b'(p - pb)}{b - pb + b'p - b'b}$
# can be rewritten as $p' = \text{expit}(\text{logit}(p) + \text{logit}(b') -
# \text{logit}(b))$ (reusing the notation of [Elkan 2001]):
#
# Starting with the right-hand side: $$p' = \text{expit}(\text{logit}(p) +
# \text{logit}(b') - \text{logit}(b))$$
#
# Recall that:
#
# - $\text{logit}(x) = \ln\left(\frac{x}{1-x}\right)$
# - $\text{expit}(x) = \frac{e^x}{1 + e^x} = \frac{1}{1 + e^{-x}}$
#
# Expand the argument of expit:
#
# $$\text{logit}(p) + \text{logit}(b') - \text{logit}(b) =
# \ln\left(\frac{p}{1-p}\right) + \ln\left(\frac{b'}{1-b'}\right) -
# \ln\left(\frac{b}{1-b}\right)$$
#
# Using logarithm properties: $$= \ln\left[\frac{p}{1-p} \times \frac{b'}{1-b'}
# \times \frac{1-b}{b}\right] = \ln\left[\frac{p \cdot b' \cdot (1-b)}{(1-p)
# \cdot (1-b') \cdot b}\right]$$
#
# Apply expit function:
#
# $$p' = \frac{1}{1 + e^{-\ln\left[\frac{p \cdot b' \cdot (1-b)}{(1-p) \cdot
# (1-b') \cdot b}\right]}}$$
#
# Since $e^{-\ln(x)} = \frac{1}{x}$: $$p' = \frac{1}{1 + \frac{(1-p) \cdot
# (1-b') \cdot b}{p \cdot b' \cdot (1-b)}}$$
#
# Multiplying numerator and denominator by $p \cdot b' \cdot (1-b)$: $$p' =
# \frac{p \cdot b' \cdot (1-b)}{p \cdot b' \cdot (1-b) + (1-p) \cdot (1-b')
# \cdot b}$$
#
# Expand the denominator:
#
# $$p \cdot b' \cdot (1-b) + (1-p) \cdot (1-b') \cdot b$$ $$= pb'(1-b) +
# (1-p)(1-b')b$$ $$= pb' - pb'b + b - pb - b'b + pb'b$$
#
# The $pb'b$ terms cancel: $$= pb' + b - pb - b'b$$ $$= b'p - pb + b - b'b$$
# $$= b - pb + b'p - b'b$$
#
# Simplify the numerator:
#
# $$p \cdot b' \cdot (1-b) = pb'(1-b) = pb' - pb'b = b'p - b'pb = b'(p - pb)$$
#
# Therefore: $$p' = \frac{b'(p - pb)}{b - pb + b'p - b'b}$$
#
# This matches exactly the formula of [Elkan 2001], completing the proof.
# $\square$
