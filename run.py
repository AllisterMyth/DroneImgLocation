import os
import subprocess
import sys
import time

INPUT_FOLDER = "./Droneimg"
BATCH_SIZE = 3
PYTHON_EXE = sys.executable 

def run_manager():
    img_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    total_images = len(img_files)
    
    if total_images == 0:
        print("No images found in Droneimg folder.")
        return

    # Reset state at the start of a brand new run
    if os.path.exists("tracker_state.json"):
        os.remove("tracker_state.json")

    print(f"Total images found: {total_images}. Processing in batches of {BATCH_SIZE}...")

    for start_idx in range(0, total_images, BATCH_SIZE):
        print(f"\n>>> LAUNCHING BATCH: Starting at image {start_idx}")
        
        # This call blocks until mainopt.py processes its 10 images and exits.
        # When it exits, the OS clears all RAM and VRAM used by it.
        try:
            subprocess.run([
                PYTHON_EXE, "main.py", 
                "--start", str(start_idx), 
                "--count", str(BATCH_SIZE)
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"Batch starting at {start_idx} encountered an error. Continuing to next batch...")
        
        # Brief pause to let the OS finalize memory cleanup
        time.sleep(1)

    print("\n[FINISH] All images processed. System memory was fully cleared between every batch.")

if __name__ == "__main__":
    run_manager()