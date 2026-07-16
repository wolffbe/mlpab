import base64

with open("data/train.csv", "rb") as f:
    train_b64 = base64.b64encode(f.read()).decode("utf-8")

with open("data/score.csv", "rb") as f:
    score_b64 = base64.b64encode(f.read()).decode("utf-8")

print("train_b64 = \"{}\"".format(train_b64))
print("score_b64 = \"{}\"".format(score_b64))