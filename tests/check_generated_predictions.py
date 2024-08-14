# %%
from pathlib import Path

path_to_predictions = Path(Path.cwd() / ".." / "python_files" / "predictions")
assert path_to_predictions.exists()

# %%
assert len(list(path_to_predictions.glob("*.npy"))) == 4
