import subprocess
import sys
import os

# Run the training script
subprocess.run([sys.executable, "train_model.py"], check=True)

# Call main if run as a module
if __name__ == "__main__":
    import train_model
    train_model.main()
