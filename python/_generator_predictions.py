# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=100_000, class_sep=0.5, random_state=0
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# %%
y_pred = model.predict_proba(X_test)[:, 1]

# %%
from sklearn.calibration import CalibrationDisplay

CalibrationDisplay.from_predictions(y_test, y_pred, strategy="uniform", n_bins=10)

# %%
from scipy.special import expit

y_pred = model.predict_proba(X_test)[:, 1]
y_pred = expit(10 * (y_pred - 0.5))

CalibrationDisplay.from_predictions(y_test, y_pred, strategy="uniform", n_bins=10)
# %%
import numpy as np

y_pred = model.predict_proba(X_test)[:, 1]
y_pred = np.log(y_pred)
y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())

CalibrationDisplay.from_predictions(y_test, y_pred, strategy="uniform", n_bins=10)

# %%
import numpy as np

y_pred = model.predict_proba(X_test)[:, 1]
y_pred = np.exp(y_pred * 5)
y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())

CalibrationDisplay.from_predictions(y_test, y_pred, strategy="uniform", n_bins=10)

# %%
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=4, n_jobs=-1).fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]

# %%
CalibrationDisplay.from_predictions(y_test, y_pred, strategy="uniform", n_bins=10)

# %%
