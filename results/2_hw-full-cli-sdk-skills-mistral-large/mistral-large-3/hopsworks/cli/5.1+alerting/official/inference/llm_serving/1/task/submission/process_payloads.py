#!/usr/bin/env python3
import json
import subprocess
import os

# Read payloads
with open('data/payloads.json', 'r') as f:
    payloads = json.load(f)

responses = []

for i, payload in enumerate(payloads):
    print(f"Processing payload {i+1}: {payload}")
    
    # Write payload to file
    payload_file = f"submission/payload_{i+1}.json"
    with open(payload_file, 'w') as f:
        json.dump({"instances": [{"text": payload}]}, f)
    
    # Invoke endpoint
    cmd = ["hops", "deployment", "predict", "scorer06901d", "--file", payload_file, "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error processing payload {i+1}: {result.stderr}")
        responses.append(None)
    else:
        try:
            response = json.loads(result.stdout)
            responses.append(response)
            print(f"Response: {response}")
        except json.JSONDecodeError:
            print(f"Failed to parse response for payload {i+1}: {result.stdout}")
            responses.append(None)

# Write answers.json
with open('submission/answers.json', 'w') as f:
    scores = [response.get("score", None) if response else None for response in responses]
    json.dump({
        "endpoint_name": "scorer06901d",
        "responses": scores
    }, f)

print("Processing complete. Results written to submission/answers.json.")