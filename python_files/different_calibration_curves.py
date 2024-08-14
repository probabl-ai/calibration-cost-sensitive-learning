# %%
# Equivalent to the magic command "%run _generate_predictions.py" but it allows this
# file to be executed as a Python script.
from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("run", "_generate_predictions.py")

# %%
import numpy as np

y_true = np.load("./predictions/y_true.npy")

# %%
