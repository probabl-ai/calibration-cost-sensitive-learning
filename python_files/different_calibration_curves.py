# %% [markdown]
#
# # Understanding calibration curves
#
# In this notebook, we explore calibration curves. We use different set of predictions
# leading to different calibration curves from which we want to build insights and
# understand the impact and meaning when it comes to our predictive models.
#
# So let's first generate some predictions. The generative process is located in the
# file `_generate_predictions.py`. This process stores the true labels and the
# predicted probability estimates of several models into the `predictions` folder.

# %%
# Equivalent to the magic command "%run _generate_predictions.py" but it allows this
# file to be executed as a Python script.
from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("run", "_generate_predictions.py")

# %% [markdown]
#
# We first load the true testing labels of our problem.

# %%
import numpy as np

y_true = np.load("./predictions/y_true.npy")
y_true

# %% [markdown]
#
# We observed that we have a binary classification problem. Now, we load different
# sets of predictions of probabilities estimated by different models.

# %%
y_proba = np.load("./predictions/y_prob_1.npy")
y_proba

# %% [markdown]
#
# We assess the calibration of the model that provide these predictions by plotting
# the calibration curve.

# %%
from sklearn.calibration import CalibrationDisplay

params = {"n_bins": 10, "strategy": "quantile"}
disp = CalibrationDisplay.from_predictions(y_true, y_proba, **params)
_ = disp.ax_.set(
    title="Calibrated model",
    xlim=(0, 1),
    ylim=(0, 1),
    aspect="equal",
)

# %%
y_proba = np.load("./predictions/y_prob_2.npy")
disp = CalibrationDisplay.from_predictions(y_true, y_proba, **params)
_ = disp.ax_.set(
    title="Uncalibrated model",
    xlim=(0, 1),
    ylim=(0, 1),
    aspect="equal",
)

# %%
y_proba = np.load("./predictions/y_prob_3.npy")
disp = CalibrationDisplay.from_predictions(y_true, y_proba, **params)
_ = disp.ax_.set(
    title="Uncalibrated model",
    xlim=(0, 1),
    ylim=(0, 1),
    aspect="equal",
)

# %%
y_proba = np.load("./predictions/y_prob_4.npy")
disp = CalibrationDisplay.from_predictions(y_true, y_proba, **params)
_ = disp.ax_.set(
    title="Uncalibrated model",
    xlim=(0, 1),
    ylim=(0, 1),
    aspect="equal",
)
# %%
