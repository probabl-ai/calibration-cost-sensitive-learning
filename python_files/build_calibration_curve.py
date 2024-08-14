# %% [markdown]
# # What is a calibration curve?
#
# Before we dive into how to interpret a calibration curve, let's start by getting
# intuitions on what it graphically represents. In this exercise, you will build your
# own calibration curve.
#
# To simplify the process, we only focus on the output of a binary classifier but
# without going into details on the model used to generate the predictions. We later
# discuss the implications of the data modeling process on the calibration curve.
#
# So let's first generate some predictions. The generative process is located in the
# file `_generate_predictions.py`. This process stores the true labels and the
# predicted probability estimates of several models into the `predictions` folder.

# %%
# equivalent to % run _generate_predictions.py
from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("run", "_generate_predictions.py")

# %% [markdown]
# Let's load a the true labels and the predicted probabilities of one of the models.

# %%
import numpy as np

y_true = np.load("./predictions/y_true.npy")
y_prob = np.load("./predictions/y_prob_2.npy")

# %%
y_true

# %%
y_prob

# %%
import pandas as pd

bin_identifier = pd.cut(y_prob, bins=10)

# %%
binned_predictions = pd.DataFrame(
    {
        "y_true": y_true,
        "y_prob": y_prob,
        "bin_identifier": bin_identifier,
    }
)
binned_predictions

# %%
import matplotlib.pyplot as plt

plt.plot(
    binned_predictions.groupby("bin_identifier").y_prob.mean(),
    binned_predictions.groupby("bin_identifier").y_true.mean(),
    "o-",
)

# %%
from sklearn.calibration import CalibrationDisplay

CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10)

# %%
