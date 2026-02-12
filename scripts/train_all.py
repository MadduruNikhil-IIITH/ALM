import subprocess
import sys
import time
import os

def run_command(command):
    print(f"Running: {command}")
    start_time = time.time()
    
    # Ensure current directory is in PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    result = subprocess.run(command, shell=True, env=env)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"Finished in {duration:.2f}s")

def main():
    print("starting ALM Full Training Pipeline...")
    print("=" * 50)
    
    # 1. Vision SFT
    print("\n[1/2] Training Vision Model (Image -> UPG)...")
    run_command(f"{sys.executable} src/sft/train_vision.py")
    
    # 2. Logic SFT
    print("\n[2/2] Training Logic Model (CoPT generation)...")
    run_command(f"{sys.executable} src/sft/train_logic.py")
    
    print("\n" + "=" * 50)
    print("All training COMPLETE!")
    print("Fine-tuned adapters should be in 'checkpoints/' and 'models/'.")

if __name__ == "__main__":
    main()
