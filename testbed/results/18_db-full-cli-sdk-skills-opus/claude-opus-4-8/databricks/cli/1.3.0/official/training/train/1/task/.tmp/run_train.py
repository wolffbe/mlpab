import os

# Run the provided, unmodified training script with the volume as the working
# directory so its relative reads/writes (train.csv, score.csv, predictions.csv)
# resolve against the uploaded data on the platform.
WORKDIR = "/Volumes/workspace/mlpaba31153/jobdata"
os.chdir(WORKDIR)

with open("train_model.py") as f:
    code = f.read()

exec(compile(code, "train_model.py", "exec"), {"__name__": "__main__"})

print("predictions.csv written:", os.path.exists(os.path.join(WORKDIR, "predictions.csv")))
