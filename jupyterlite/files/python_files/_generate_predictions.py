# %%
from pathlib import Path

Path.mkdir(Path.cwd() / ".." / "predictions", exist_ok=True)

# %%
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=100_000, class_sep=0.5, random_state=0
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
np.save("../predictions/y_true.npy", y_test)

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# %%
raw_y_pred = model.predict_proba(X_test)[:, 1]

# %%
np.save("../predictions/y_prob_1.npy", raw_y_pred)

# %%
from scipy.special import expit

y_pred = expit(10 * (raw_y_pred - 0.5))

np.save("../predictions/y_prob_2.npy", y_pred)


# %%
from scipy.interpolate import interp1d

# Mirror the previous sigmoid curve around the diagonal:
y_pred = interp1d(y_pred, raw_y_pred, fill_value="extrapolate")(raw_y_pred)
y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())

np.save("../predictions/y_prob_3.npy", y_pred)

# %%
# Equality reweighting the classes at training time.

p = raw_y_pred
w = 0.3
y_pred = (p / w) / (1 - p + p / w)

np.save("../predictions/y_prob_4.npy", y_pred)

