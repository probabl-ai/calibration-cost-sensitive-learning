# %% [markdown]
#
# # Imbalanced classification: pitfalls and solutions
#
# Imbalanced classification refers to issues where the balance of the class frequencies
# in the target variable creates additional challenges for the classification problem.
# We focus on two particular issues related to imbalanced classification.
#
# The first issue is related to a large difference between the class frequencies in the
# target variable. It means that the event of interest to predict is rare. As an
# example, in fraud detection, the event of interest is a fraud and is much less common
# than legitimate transactions. In this notebook, we first focus on studying this use
# case that researchers have not addressed properly in the scientific literature except
# in recent works.
#
# The second issue is related to the fact that the data acquisition process do not
# reflect the true class balance. This means that the class frequencies in the target
# variable are not representative of the true class balance. As an example, for medical
# diagnosis, the data acquisition process may be biased towards patients with a rare
# disease by collecting equal numbers of patients with the disease and equal numbers of
# patients without the disease. Therefore, there is a need to correct this bias. In the
# next notebook, we will focus on this issue.
#
# ## Class imbalance: representative data acquisition with rare events of interest
#
# In real-world applications, we commonly need to predict rare events, e.g. frauds, rare
# diseases, rare climatic events, etc. Simplifying this problem to a binary outcome, it
# means that the probability for this rare event to happen is rather low in comparison
# to the probability of the rare event not to happen.
#
# To cover the implications of class imbalance, we first generate a synthetic dataset
# for which we control the success rate of the positive class. We define the generative
# process below as follows:
#
# - We generate a vector of coefficients `true_coef` of shape `(n_features,)` where each
#   element is a standard normal random variable. In short, it is the true model that we
#   would like to learn.
# - We generate a matrix of features `X` of shape `(n_samples, n_features)` where each
#   column is a standard normal random variable.
# - We compute the linear predictor `Z` as the dot product of the features and the
#   vector of coefficients `true_coef`.
# - We transform the linear predictor `Z` into class probabilities using the sigmoid
#   function. To create rare positive events, we shift the intercept of the sigmoid
#   function.
# - Finally, we generate a binary target variable `y` where we sample each event by
#   drawing a sample from a binomial distribution with `n=1` and `p` being the
#   probability of the positive class we previously computed.

# %%
import matplotlib.pyplot as plt  # needed for pandas in jupyterlite.
import numpy as np
import pandas as pd
from scipy.special import expit

rng = np.random.default_rng(0)
n_samples, n_features = 1_000_000, 5

true_coef = rng.normal(size=n_features)
X = rng.normal(size=(n_samples, n_features))
Z = X @ true_coef
intercept = -4
y = rng.binomial(n=1, p=expit(Z + intercept))

# create pandas data structures for convenience
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
y = pd.Series(y, name="target")

# %% [markdown]
#
# Let's look at the true target and especially the class frequencies and absolute
# counts.

# %%
print(f"Class frequencies:\n {y.value_counts(normalize=True) * 100}")
print(f"Class counts:\n {y.value_counts()}\n")

# %% [markdown]
#
# Looking at the true target distribution, we therefore observe that the probability for
# a sample to be the positive class with label 1 is rare (~2.5%). Regarding absolute
# counts, because we generated 1,000,000 samples, the number of events of interest is
# rather high (25,000).
#
# A particular challenge when dealing with real-world class imbalance is that the number
# of available samples of the rare event can be usually low even with a large number of
# samples. Therefore, it is always important to check the absolute counts of the rare
# event and if the dataset contains less than 1,000 samples of the rare event, then you
# are exactly in the same situation as having a dataset with a low number of samples
# with all related challenges (e.g. large variance of the estimator, weak signal, etc.).
#
# ## Learning a predictive model
#
# Here, we know that our generative process is a linear relationship between the
# features and the target. Therefore, a linear predictive model is perfectly suited to
# learn the true model. We therefore train a logistic regression model.

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty=None).fit(X, y)

# %% [markdown]
#
# Let's check if the learned model is able to recover the true model.

# %%
comparison_coef = pd.DataFrame(
    {
        "Data generating model": np.hstack((intercept, true_coef)),
        "Unpenalized logistic regression": np.hstack(
            (model.intercept_, model.coef_.flatten())
        ),
    },
    index=np.hstack(["intercept", model.feature_names_in_]),
)
ax = comparison_coef.plot.barh()
_ = ax.set(
    title="Comparison of the true and learned model coefficients",
    xlabel="Coefficient value",
    ylabel="Feature",
)

# %% [markdown]
#
# We observe that the learned model is able to recover the true model coefficients.
# However, be aware that it is not necessarily the case. Let's do a small exercise that
# illustrates when one of the assumptions to recover the true model is not met.
#
# ### Exercise
#
# Below, write a small function that embeds the generative process that we defined
# above. This time only generate 10,000 samples, train a logistic regression model and
# check the learned model coefficients.
#
# Do you recover the true model coefficients? If not, what is the reason?

# %%

# %% [markdown]
#
# ### Solution


# %%
def generate_imbalanced_dataset(n_samples=10_000, n_features=5, seed=0):
    rng = np.random.default_rng(seed)

    true_coef = rng.normal(size=n_features)
    X = rng.normal(size=(n_samples, n_features))
    Z = X @ true_coef
    intercept = -4
    y = rng.binomial(n=1, p=expit(Z + intercept))

    # create pandas data structures for convenience
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y, name="target")

    return X, y


X_exercise, y_exercise = generate_imbalanced_dataset(n_samples=10_000)
model_exercise = LogisticRegression(penalty=None).fit(X_exercise, y_exercise)

comparison_coef_exercise = pd.DataFrame(
    {
        "Data generating model": np.hstack((intercept, true_coef)),
        "Unpenalized logistic regression": np.hstack(
            (model_exercise.intercept_, model_exercise.coef_.flatten())
        ),
    },
    index=np.hstack(["intercept", model_exercise.feature_names_in_]),
)
ax = comparison_coef_exercise.plot.barh()
_ = ax.set(
    title=(
        "Comparison of the true and learned model coefficients\n"
        "trained on a smaller dataset"
    ),
    xlabel="Coefficient value",
    ylabel="Feature",
)

# %% [markdown]
#
# We observe that we have a larger difference between the coefficients of the true
# generative process and the learned model. The reason is that the coefficients of the
# generative process can be recovered under the following assumptions:
#
# - An infinite number of samples is available. Therefore, with a larger number of
#   samples, the coefficients of the predictive model will get closer to the true
#   coefficients.
# - The predictive model should be well specified. In other words, if our predictive
#   model is not flexible enough then it will underfit and not recover all the signal of
#   the true model.
#
# We need to study an additional assumption regarding the probabilities estimated by our
# predictive model.

# %%
y_proba = model.predict_proba(X)
y_proba = pd.DataFrame(y_proba, columns=["p_hat(y=0)", "p_hat(y=1)"])
y_proba

# %% [markdown]
#
# Our predictive model estimates the probabilities of the class of interest (i.e.
# `p_hat(y=1)`). However, we estimate those numbers and they do not necessarily reflect
# the true probabilities. Here, we can compute the mean of the estimated probabilities
# and check if we are close to the true probability of the positive class.

# %%
y_proba.mean() * 100

# %%
_ = y_proba.plot.hist(
    bins=100, figsize=(10, 5), subplots=True, layout=(1, 2), sharey=True
)

# %%
bins = np.linspace(0, 1, 300)
_ = (
    pd.concat([y_proba, y.to_frame()], axis=1)
    .groupby("target")["p_hat(y=1)"]
    .plot.hist(bins=bins, alpha=0.5, legend=True, density=True)
)

# %% [markdown]
#
# For our example, we are indeed close to the true frequency. The reason is that the
# algorithm used `LogisticRegression` minimizes a "strictly proper" scoring rule. If we
# do not use such a loss function, there are no theoretical guarantees that the
# estimated probabilities will be close to the true probabilities.
#
# To conclude, the three above conditions work together: the strictly proper scoring
# rule provides the right objective, the well-specified model ensures the true
# coefficients exist within the model's parameter space, and infinite samples allow the
# optimization to converge to the global optimum that corresponds to these true
# coefficients.
#
# With these information, we would expect our classifier to be well calibrated. We can
# check that by plotting the calibration curve.

# %%
from sklearn.calibration import CalibrationDisplay

display = CalibrationDisplay.from_estimator(model, X, y, n_bins=10, strategy="quantile")
_ = display.ax_.set_title("Calibration curve of the unpenalized logistic regression")

# %% [markdown]
#
# Since we have rare events, only a few samples have high probability and the
# quantile-based strategy will not show a curve on the right side of the plot. Let's
# zoom in on the plot to see the curve.

# %%
display.plot()
axis_lim = (
    min(display.prob_true.min(), display.prob_pred.min()) * 0.9,
    max(display.prob_true.max(), display.prob_pred.max()) * 1.1,
)
_ = display.ax_.set(
    xlim=axis_lim,
    ylim=axis_lim,
    title="Calibration curve of the unpenalized logistic regression",
)

# %% [markdown]
#
# We observe that the logistic regression is well calibrated as the curve is close to
# the diagonal line. It therefore means that the probabilities estimated by the model
# are close to the true probabilities.
#
# ## From probabilities to predicted outcomes (and to operational decisions)
#
# Up to this point of the notebook, we have not encountered any real issues due to the
# fact that our dataset is imbalanced: with enough samples, a well-specified model
# minimizing a strictly proper scoring rule, everything seems to be fine.
#
# However, practitioners have been complaining for many years regarding the above
# setting. Indeed, the issue comes from when one seeks to translate the estimated
# probabilities into predicted classification outcomes.
#
# In classification, the predicted outcomes correspond to the classes of the target. As
# a general rule, the estimated probabilities of the classifier are processed to provide
# a single outcome for each sample. In general the most probable class is selected. For
# binary classification, it means that the probability is thresholded with a decision
# cut-off value set at 0.5. In scikit-learn, it corresponds to the `predict` method.
# Let's check the link between the `predict_proba` and `predict` methods.

# %%
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

np.allclose(y_pred, y_proba[:, 1] > 0.5)

# %% [markdown]
#
# Predicted outcomes come with a set of metrics. We derive those metrics from the
# confusion matrix indicating the number of true positives, true negatives, false
# positives and false negatives.

# %%
from sklearn.metrics import ConfusionMatrixDisplay

display = ConfusionMatrixDisplay.from_predictions(y, y_pred)
_ = display.ax_.set_title("Confusion matrix of the unpenalized logistic regression")

# %% [markdown]
#
# From the confusion matrix above, we can already understand what bothers practitioners
# in practice: the number of true positives and thus the number of rare events detected
# is zero.
#
# One could interpret that our model is therefore not able to detect rare events and
# thus useless. In general, instead of using the confusion matrix, practitioners use
# different metrics such as the precision, recall, etc. Let's check the classification
# report available in scikit-learn that provides a summary of the metrics.

# %%
from sklearn.metrics import classification_report

print(classification_report(y, model.predict(X)))

# %% [markdown]
#
# As expected, the precision and recall for the class of interest is null.
#
# In the next section, we present the usual solutions used by practitioners to deal with
# this problem.
#
# ## What people naively do and why you should not do it
#
# The reason for not having any true positives in the confusion matrix is that the
# estimated probabilities by the model for rare events are low because as previously
# shown, those events are rare!
#
# One way to counter this issue is to resample the dataset and balance the class
# frequencies. This means that we artificially increase the number of samples of the
# rare event and thus the likelihood of the rare event to be detected is higher. We
# therefore boost the estimated probabilities related to those rare events.
#
# Let's use `imbalanced-learn` to resample the dataset before training a logistic
# regression model.

# %%
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

# keep a 0.7 ratio between the number of samples of the rare event and the number of
# samples of the majority event.
model = make_pipeline(
    RandomUnderSampler(sampling_strategy=0.7, random_state=0),
    LogisticRegression(penalty=None),
).fit(X, y)

# %% [markdown]
#
# Now, let's repeat the previous experiment and check the confusion matrix and the
# classification report.

# %%
display = ConfusionMatrixDisplay.from_estimator(model, X, y)
_ = display.ax_.set_title("Confusion matrix of the under-sampled logistic regression")

# %%
print(classification_report(y, model.predict(X)))

# %% [markdown]
#
# We observe that the number of true positives is now non-zero as well as the precision
# and recall for the class of interest.
#
# So we might be tempted to conclude that we did the right thing by resampling the
# dataset. However, here we only looked at the "thresholded" metrics. We should study
# the calibration of the model and we can have a look at the coefficients also.
#
# ### Exercise
#
# Plot the coefficients of the model and check whether or not the coefficients are close
# to the true coefficients. Then, plot the calibration curve and check whether or not
# the model is well calibrated. What do you observe?

# %%


# %% [markdown]
#
# ### Solution

# %%
comparison_coef = pd.DataFrame(
    {
        "Data generating model": np.hstack((intercept, true_coef)),
        "Model trained on under-sampled data": np.hstack(
            (model[-1].intercept_, model[-1].coef_.flatten())
        ),
    },
    index=np.hstack(["intercept", model.feature_names_in_]),
)
ax = comparison_coef.plot.barh()
_ = ax.set(
    title="Comparison of the true and learned model coefficients",
    xlabel="Coefficient value",
    ylabel="Feature",
)

# %%
display = CalibrationDisplay.from_estimator(
    model,
    X,
    y,
    n_bins=20,
    strategy="quantile",
    name="Model trained on under-sampled data",
)
display.ax_.set_title(
    "Calibration curve of the under-sampled logistic regression"
)
_ = display.ax_.legend(loc="upper right")

# %% [markdown]
#
# We observe that the coefficients related to the features are close to the true
# coefficients of the generative model. However, the intercept is completely off. It
# translates into an uncalibrated model as seen in the calibration curve: our model
# becomes too confident at predicting the rare event which is not surprising because it
# is exactly what we were seeking for.
#
# ### Exercise
#
# Since our model is not well calibrated, as an exercise, re-calibrate the model using
# the `sklearn.calibration.CalibratedClassifierCV` and check both the calibration curve,
# the confusion matrix, and the classification report. In the `CalibratedClassifierCV`
# set the parameter `method="isotonic"`.
#
# What do you observe?

# %%
from sklearn.calibration import CalibratedClassifierCV

# %% [markdown]
#
# ### Solution

# %%
calibrated_model = CalibratedClassifierCV(model, method="isotonic")
calibrated_model.fit(X, y)

# %%
display = CalibrationDisplay.from_estimator(
    calibrated_model, X, y, n_bins=20, strategy="quantile"
)

axis_lim = (
    min(display.prob_true.min(), display.prob_pred.min()) * 0.9,
    max(display.prob_true.max(), display.prob_pred.max()) * 1.1,
)
_ = display.ax_.set(
    xlim=axis_lim,
    ylim=axis_lim,
    title="Calibration curve of the calibrated under-sampled logistic regression",
)

# %%
display = ConfusionMatrixDisplay.from_estimator(calibrated_model, X, y)
_ = display.ax_.set_title(
    "Confusion matrix of the calibrated under-sampled logistic regression"
)

# %%
print(classification_report(y, calibrated_model.predict(X)))

# %% [markdown]
#
# So in terms of calibration, we see that the `CalibratedClassifierCV` is able to
# calibrate the model. When looking at the confusion matrix, and the classification
# report, we see that we reverted the effect of the resampling and we are back to square
# one.
#
# So what is the lesson to learn here?
#
# Resampling acts by artificially shifting the class distribution such that rare events
# are more likely during the training process. It impacts the predicted outcomes and for
# the simple case where we have a well-defined linear model, it is equivalent to
# shifting the intercept. However, the estimated probabilities are completely off the
# original true probabilities.
#
# Therefore, it tells us that we should be careful with the choice of evaluation metrics
# and how it interacts with the choice of the decision cut-off threshold.
#
# Ranking metrics (e.g. ROC AUC) and probabilistic metrics (e.g. log loss) that assess
# both ranking and calibration of the predictive model at the same time are good choices
# but do not reflect on the choice of the decision cut-off threshold.
#
# "Thresholded" metrics (e.g. precision, recall) are impacted by the decision cut-off
# threshold. Therefore, looking such metrics only for a single decision cut-off
# is not informative enough. It is required to look at those metrics by varying the
# decision cut-off threshold.
#
# The next section focuses on setting the decision cut-off threshold when the evaluation
# metric of interest is a "thresholded" metric.
#
# ## Choosing the decision cut-off threshold when "thresholded" metrics are used
#
# In this section, we show two useful meta-estimators available in scikit-learn to set
# the decision cut-off threshold to change the predicted outcomes of a classifier.
#
# On the one hand, the `FixedThresholdClassifier` meta-estimator accepts an explicit
# value that is used to threshold the estimated probabilities into predicted outcomes.
# The value is defined by the user and is not optimized to maximize a specific metric.
#
# On the other hand, the `TunedThresholdClassifierCV` meta-estimator tunes the decision
# cut-off threshold to maximize a specific metric. The metric is defined by the user and
# is optimized using cross-validation.
#
# Let's first demonstrate how to use the `FixedThresholdClassifier` meta-estimator.
# First, let's define a vanilla logistic regression model since we previously saw that
# the resulting model is well calibrated when fitted on the original dataset.

# %%
model = LogisticRegression(penalty=None).fit(X, y)

# %% [markdown]
#
# Now, let's say that we would like to get a model with a specific precision-recall
# trade-off. For such analysis, we compute the precision and recall as a function of
# the decision cut-off threshold as well as the precision-recall curve.

# %%
import numpy as np
from sklearn.metrics import make_scorer, precision_score, recall_score

# The following functionality is not yet implemented in scikit-learn and we use a bit
# of private API to easily compute the precision and recall as a function of the
# decision cut-off threshold. In the future, you can refer to the following PR that
# implements such functionality:
# https://github.com/scikit-learn/scikit-learn/pull/31338
from sklearn.metrics._scorer import _CurveScorer as CurveScorer

thresholds = np.linspace(0, 1, 100)
precision_curve_scorer = CurveScorer.from_scorer(
    make_scorer(precision_score, zero_division=0),
    response_method="predict_proba",
    thresholds=thresholds,
)
recall_curve_scorer = CurveScorer.from_scorer(
    make_scorer(recall_score, zero_division=0),
    response_method="predict_proba",
    thresholds=thresholds,
)

# %%
precision_scores, precision_thresholds = precision_curve_scorer(model, X, y)
recall_scores, recall_thresholds = recall_curve_scorer(model, X, y)

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig_plotly = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Precision and Recall vs Threshold", "Precision-Recall Curve"),
    horizontal_spacing=0.1,
)

fig_plotly.add_trace(
    go.Scatter(
        x=precision_thresholds,
        y=precision_scores,
        mode="lines+markers",
        name="Precision",
        marker=dict(symbol="cross"),
        hovertemplate="Threshold: %{x:.2f}<br>Precision: %{y:.3f}",
    ),
    row=1,
    col=1,
)
fig_plotly.add_trace(
    go.Scatter(
        x=recall_thresholds,
        y=recall_scores,
        mode="lines+markers",
        name="Recall",
        marker=dict(symbol="cross"),
        hovertemplate="Threshold: %{x:.2f}<br>Recall: %{y:.3f}",
    ),
    row=1,
    col=1,
)

fig_plotly.add_trace(
    go.Scatter(
        x=recall_scores,
        y=precision_scores,
        mode="lines+markers",
        name="PR Curve",
        marker=dict(symbol="circle"),
        hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<br>Threshold: %{text}",
        text=[f"{t:.2f}" for t in precision_thresholds],
        showlegend=False,
    ),
    row=1,
    col=2,
)

fig_plotly.update_layout(
    legend=dict(
        x=0.35,
        y=0.85,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
    ),
    hovermode="closest",
    width=1200,
    height=500,
)

fig_plotly.update_xaxes(title_text="Threshold", range=[0, 1], row=1, col=1)
fig_plotly.update_yaxes(title_text="Score", range=[0, 1], row=1, col=1)
fig_plotly.update_xaxes(title_text="Recall", range=[0, 1], row=1, col=2)
fig_plotly.update_yaxes(title_text="Precision", range=[0, 1], row=1, col=2)
fig_plotly.show()

# %% [markdown]
#
# Using these curves, we now can make a choice regarding a specific trade-off between
# the level of recall and precision for our classifier. Let's expose a possible use
# case: our model could be used for medical diagnosis and in this particular setting,
# we could imagine that physicians reviewing cases of rare diseases expect a certain
# level of precision of the computer-aided diagnosis system. Otherwise, the system
# will show too many false positive cases, tiring the physicians, leading to potential
# errors. However, while expecting a certain level of precision, we also would like our
# computer-aided diagnosis system to maximize the recall level.
#
# Thus, by looking at the precision-recall curve above, we could impose a minimum level
# of precision of 10%. It means that it would define an horizontal line at 0.05 on the
# y-axis. We will consider all points above this line and seek for the maximum recall
# and deduce the corresponding threshold that is 0.07.
#
# ### Exercise
#
# Using the `FixedThresholdClassifier` meta-estimator, set the decision cut-off
# threshold to the value for which the precision and recall curve intersect (i.e.
# ~0.07). Check the calibration curve, the confusion matrix, and the classification
# report.
#
# Is the model the resulting model well calibrated? What are the level of precision and
# recall of the class of interest?

# %%
from sklearn.model_selection import FixedThresholdClassifier

# %% [markdown]
#
# ### Solution

# %%
threshold = 0.07
model = FixedThresholdClassifier(
    LogisticRegression(penalty=None), threshold=threshold
).fit(X, y)

# %%
display = CalibrationDisplay.from_estimator(model, X, y, n_bins=20, strategy="quantile")

axis_lim = (
    min(display.prob_true.min(), display.prob_pred.min()) * 0.9,
    max(display.prob_true.max(), display.prob_pred.max()) * 1.1,
)
_ = display.ax_.set(
    xlim=axis_lim,
    ylim=axis_lim,
    title="Calibration curve of the fixed threshold logistic regression",
)

# %%
display = ConfusionMatrixDisplay.from_estimator(model, X, y)
_ = display.ax_.set_title(
    "Confusion matrix of the fixed threshold logistic regression"
)

# %%
print(classification_report(y, model.predict(X)))

# %% [markdown]
#
# As expected, we observe that the model is well calibrated because modifying the
# decision cut-off threshold does not impact the calibration of the model.
#
# With the selected threshold, we expect to have a minimum level of precision of 10%
# which is exactly what we observe.
#
# While it is an interesting exercise, setting the threshold manually is not the best
# practice. It would be better to use the `TunedThresholdClassifierCV` meta-estimator to
# tune the decision cut-off threshold to maximize a specific metric or a specific
# trade-off using cross-validation.
#
# Below, we show a case where we want to maximize the recall score but such that the
# model reach a minimum precision score. We therefore need to create a custom function
# that can be used by the `TunedThresholdClassifierCV` meta-estimator.


# %%
def maximize_recall_under_constrained_precision(y_true, y_pred, precision_level):
    precision, recall = precision_score(y_true, y_pred), recall_score(y_true, y_pred)

    if precision < precision_level:
        # under a certain precision level, we cannot accept the model and thus return
        # the worst possible score.
        return -np.inf
    return recall


# %%
from sklearn.model_selection import TunedThresholdClassifierCV

# create a scorer that maximizes the recall but such that the precision is at least 0.1
scoring = make_scorer(maximize_recall_under_constrained_precision, precision_level=0.1)
model = TunedThresholdClassifierCV(
    estimator=LogisticRegression(penalty=None), scoring=scoring, n_jobs=-1
).fit(X, y)

# %%
display = ConfusionMatrixDisplay.from_estimator(model, X, y)
_ = display.ax_.set_title(
    "Confusion matrix of the tuned threshold logistic regression"
)

# %%
print(classification_report(y, model.predict(X)))

# %% [markdown]
#
# Looking at the confusion matrix, we observe that we detect a certain number of rare
# events. Looking into the classification report, we observe that the constraint set on
# the precision is respected (i.e. the precision is 0.11). For such precision, the
# maximum recall is 0.15. Now, let's check what is the decision cut-off threshold that
# was found during the cross-validation procedure.

# %%
float(model.best_threshold_)

# %% [markdown]
#
# Here, we chose to maximize a specific metric under a constraint. It is the best choice
# when no "business" metric is known for the machine learning task at hand. However, be
# aware that if you have a "business" metric available, then you should use it together
# with the `TunedThresholdClassifierCV` meta-estimator. To see an example, refer to the
# notebook entitled "Cost-sensitive learning to optimize a business metrics" from this
# course.
