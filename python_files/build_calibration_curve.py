# %% [markdown]
#
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
# Equivalent to the magic command "%run _generate_predictions.py" but it allows this
# file to be executed as a Python script.
from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("run", "_generate_predictions.py")

# %% [markdown]
#
# Let's load a the true labels and the predicted probabilities of one of the models.

# %%
import numpy as np

y_true = np.load("./predictions/y_true.npy")
y_prob = np.load("./predictions/y_prob_2.npy")

# %%
y_true

# %% [markdown]
#
# `y_true` contains the true labels. In this case, we face a binary classification
# problem.

# %%
y_prob

# %% [markdown]
#
# `y_prob` contains the predicted probabilities of the positive class estimated by a
# given model.
#
# As discussed earlier, we could evaluate the discriminative power of the model using
# a metric such as the ROC-AUC score.

# %%
from sklearn.metrics import roc_auc_score

print(f"ROC AUC: {roc_auc_score(y_true, y_prob):.2f}")

# %% [markdown]
#
# The score is much above 0.5 and it means that our model is able to discriminate
# between the two classes. However, this score does not tell us if the predicted
# probabilities are well calibrated.
#
# We can provide an example of a well-calibrated probability: for a given sample, the
# predicted probabilities is considered well-calibrated if the proportion of samples
# where this probability is predicted is equal to the probability itself.
#
# It is therefore possible to group predictions into bins based on their predicted
# probabilities and to calculate the proportion of positive samples in each bin and
# compare it to the average predicted probability in the bin.
#
# So let's create a calibration curve following the steps below:
#
# 1. Bin the predicted probabilities (i.e. `y_prob`) into 10 bins. You can use the
#    `pd.cut` function from the `pandas` library. It will return a `Categorical`
#    object where we get the bin identifier for each sample.
# 2. Create a DataFrame with the true labels, predicted probabilities, and bin
#    identifier.
# 3. Group the DataFrame by the bin identifier and calculate the mean of the true
#    labels and the predicted probabilities.
# 4. Plot the calibration curve by plotting the average predicted probabilities
#    against the fraction of positive samples.

# %%
import pandas as pd

n_bins = 10
bin_identifier = pd.cut(y_prob, bins=n_bins)

# %%
predictions = pd.DataFrame(
    {
        "y_true": y_true,
        "y_prob": y_prob,
        "bin_identifier": bin_identifier,
    }
)
predictions

# %%
avg_predicted_probabilities = predictions.groupby(
    "bin_identifier", observed=True
).y_prob.mean()
fraction_positive_samples = predictions.groupby(
    "bin_identifier", observed=True
).y_true.mean()

# %%
import matplotlib.pyplot as plt

_, ax = plt.subplots()
ax.plot(avg_predicted_probabilities, fraction_positive_samples, "o-")
_ = ax.set(
    xlim=(0, 1),
    ylim=(0, 1),
    xlabel="Average predicted probability",
    ylabel="Fraction of positive samples",
    aspect="equal",
    title="Calibration curve",
)

# %% [markdown]
#
# Scikit-learn provides a utility to plot the calibration curve. This utility is
# available in the class `sklearn.calibration.CalibrationDisplay`. It provides a couple
# of more options such as different strategies to bin the predicted probabilities.

# %%
from sklearn.calibration import CalibrationDisplay

disp = CalibrationDisplay.from_predictions(
    y_true, y_prob, n_bins=n_bins, strategy="quantile"
)
_ = disp.ax_.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
