"""Self-contained training script with embedded CSVs."""
import pandas as pd
import numpy as np
import io

# Embedded CSVs
train_csv = """row_id,f1,f2,f3,f4,f5,label
T00000,-0.532466,0.71541,0.540049,0.530937,-0.305125,0
T00001,-0.58163,-0.26598,-0.536864,0.655941,-2.033532,0
T00002,0.961031,-1.194168,-0.196191,-1.520891,1.057815,1
T00003,-0.730457,-1.296813,-1.572949,1.261362,-0.736444,0
"""

score_csv = """row_id,f1,f2,f3,f4,f5
S00000,-0.409212,-0.983899,-1.387183,-0.516577,-0.517954
S00001,1.164751,-0.330792,-0.990752,0.839924,0.32402
S00002,0.324428,-0.946375,0.710071,-1.124528,-0.332173
S00003,-1.523611,-0.170416,1.677581,1.651235,-0.158976
"""

# Write CSVs to files
with open("train.csv", "w") as f:
    f.write(train_csv)

with open("score.csv", "w") as f:
    f.write(score_csv)

# Original training script
FEATURES = ["f1", "f2", "f3", "f4", "f5"]
LEARNING_RATE = 0.1
ITERATIONS = 300

def main():
    train = pd.read_csv("train.csv")
    score = pd.read_csv("score.csv")
    X = train[FEATURES].to_numpy(dtype=float)
    y = train["label"].to_numpy(dtype=float)
    
    # Initialize weights and bias
    w = np.zeros(X.shape[1])
    b = 0.0
    
    # Gradient descent
    for _ in range(ITERATIONS):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w = w - LEARNING_RATE * (X.T @ g) / len(y)
        b = b - LEARNING_RATE * g.mean()
    
    # Predict
    Xs = score[FEATURES].to_numpy(dtype=float)
    preds = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
    out = pd.DataFrame({"row_id": score["row_id"], "score": np.round(preds, 6)})
    out.to_csv("predictions.csv", index=False)
    
    # Write to DBFS for access
    with open("/dbfs/FileStore/{}/predictions.csv".format("mlpaba53a68"), "w") as f:
        out.to_csv(f, index=False)

if __name__ == "__main__":
    main()