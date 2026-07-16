import math
import json

def score(text):
    A = 1.804951
    B = 0.883156
    C = 0.962866
    D = 0.353537
    
    def _trigram_weight(tri):
        o0, o1, o2 = ord(tri[0]), ord(tri[1]), ord(tri[2])
        return math.sin(A * o0 + B * o1 + C * o2 + D)
    
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    
    return {"score": round(ll, 6)}

# Test the function with our payloads
payloads = [
    "store lookup stream training batch online online embedding monitor store model",
    "pipeline inference store embedding training latency vector inference latency serving vector",
    "lookup training batch registry vector latency online",
    "schedule schedule stream drift batch pipeline training training store stream model inference",
    "stream store latency batch pipeline registry training feature store batch"
]

responses = []
for payload in payloads:
    responses.append(score(payload))

# Write the responses to the submission file
result = {
    "endpoint_name": "scorerb7352a",
    "responses": responses
}

with open("submission/answers.json", "w") as f:
    json.dump(result, f, indent=2)

print("Responses:")
for i, response in enumerate(responses):
    print(f"Payload {i+1}: {response}")