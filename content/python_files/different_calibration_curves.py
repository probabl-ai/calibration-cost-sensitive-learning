# %% [markdown]
#
# # Reading calibration curves
#
# In this notebook, we explore calibration curves. We use different set of
# predictions leading to different calibration curves from which we want to
# build insights and understand the impact and meaning when it comes to our
# predictive models.
#
# So let's first collect different prediction sets for the same classification
# task. This is achieved by a script named `_generate_predictions.py`. This
# script stores the true labels and the predicted probability estimates of
# several models into the `predictions` folder. We don't need to understand
# what model they correspond to, we just want to analyze the calibration of
# these models.

# %%
# Equivalent to the magic command "%run _generate_predictions.py" but it allows this
# file to be executed as a Python script.
from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("run", "../python_files/_generate_predictions.py")

# %% [markdown]
#
# We first load the true testing labels of our problem.

# %%
import numpy as np

y_true = np.load("../predictions/y_true.npy")
y_true

# %%
unique_class_labels, counts = np.unique(y_true, return_counts=True)
unique_class_labels

# %%
counts

# %% [markdown]
#
# We observe that we have a binary classification problem. Now, we load different
# sets of predictions of probabilities estimated by different models.

# %%
y_proba_1 = np.load("../predictions/y_prob_1.npy")
y_proba_1

# %% [markdown]
#
# We assess the calibration of the model that provide these predictions by
# plotting the calibration curve:
# 
# - data points are first **grouped into bins of similar predicted
#   probabilities**;
# - then for each bin, we plot a point of the curve that represents the
#   **fraction of observed positive labels in a bin** against the **mean
#   predicted probability for the positive class in that bin**.

# %%
from sklearn.calibration import CalibrationDisplay

params = {"n_bins": 10, "strategy": "quantile"}
disp = CalibrationDisplay.from_predictions(y_true, y_proba_1, **params)
_ = disp.ax_.set(
    title="Model 1 - well calibrated",
    xlim=(0, 1),
    ylim=(0, 1),
    aspect="equal",
)

# %% [markdown]
#
# We observe that the calibration curve is close to the diagonal that represents a
# perfectly calibrated model. It means that relying on the predicted probabilities
# will provide reliable estimates of the true probabilities.

# %% [markdown]
#
# So in addition of being well calibrated, our model is capable of distinguishing
# between the two classes.
#
# We now repeat the same analysis for the other sets of predictions.

# %%
y_proba_2 = np.load("../predictions/y_prob_2.npy")


# %%
disp = CalibrationDisplay.from_predictions(y_true, y_proba_2, **params)
disp.ax_.axvline(0.5, color="tab:orange", label="Estimated probability = 0.5")
disp.ax_.legend()
_ = disp.ax_.set(
    title="Model 2 - over confident",
    xlim=(0, 1),
    ylim=(0, 1),
    aspect="equal",
)

# %% [markdown]
#
# We added a vertical line at a 0.5 threshold for the mean predicted
# probability.
#
# Let's first focus on the **right part of the curve**, that is when the models
# predicts the positive class, assuming a decision threshold at 0.5. The
# calibration curve is below the diagonal. It means that the fraction of
# observed positive data points is lower than the predicted probabilities.
# Therefore, our model over-estimates the probabilities of the positive class
# when the predictions are higher than the default threshold: the model is
# therefore **over-confident in predicting the positive class**.
# 
# Let's now focus on the **left part of the curve**, that is when the model
# predicts the negative class. The curve is above the diagonal, meaning that
# the fraction of observed positive data points is higher than the predicted
# probabilities of the positive class. This also means that the fraction of
# observed negatives is lower than the predicted probabilities of the negative
# class. Therefore, our model is also **over-confident in predicting the
# negative class**.
#
# In conclusion, our model is over-confident when predicting either classes:
# the predicted probabilities are too close to 0 or 1 compared to the observed
# fraction of positive data points in each bin.
#
# Let's use the same approach to analyze some other typical calibration curves.

# %%
y_proba_3 = np.load("../predictions/y_prob_3.npy")
disp = CalibrationDisplay.from_predictions(y_true, y_proba_3, **params)
disp.ax_.axvline(0.5, color="tab:orange", label="Estimated probability = 0.5")
disp.ax_.legend()
_ = disp.ax_.set(
    title="Model 3 - under confident",
    xlim=(0, 1),
    ylim=(0, 1),
    aspect="equal",
)

# %% [markdown]
#
# Here, we observe the opposite behaviour compared to the previous case: our model
# output probabilities that are too close to 0.5 compared to the empirical positive
# class fraction. Therefore, this model is under-confident.
#
# Let's check the last set of predictions.

# %%
y_proba_4 = np.load("../predictions/y_prob_4.npy")
disp = CalibrationDisplay.from_predictions(y_true, y_proba_4, **params)
disp.ax_.axvline(0.5, color="tab:orange", label="Estimated probability = 0.5")
disp.ax_.legend()
_ = disp.ax_.set(
    title="Model 4 - ",
    xlim=(0, 1),
    ylim=(0, 1),
    aspect="equal",
)

# %% [markdown]
#
# Here, we observe the opposite behaviour compared to the previous case: our model
# output relatively low probabilities while the fraction of positive samples is high.
# Therefore, this model is under-confident.
