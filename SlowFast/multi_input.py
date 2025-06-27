import os
import csv
from model import predict_video

# —— USER CONFIG ——  
VIDEO_DIR   = "/research/lfcats/data/original/video_robustness/kinetics-dataset/k400/test"
INPUT_FILE  = "/research/lfcats/projects/video_robustness/action_robustness/SlowFast/inputs.csv"
OUTPUT_FILE = "/research/lfcats/projects/video_robustness/action_robustness/SlowFast/outputs.csv"

# Set to an integer to process only that many files, or to None to process all.
N = 5

def run_inference(video_dir: str, input_csv: str, output_csv: str, n: int = None):
    # 1) Read all the filenames
    with open(input_csv, newline="") as f:
        reader = csv.reader(f)
        filenames = [row[0] for row in reader]

    # 2) Optionally trim to first n
    if n is not None:
        filenames = filenames[:n]

    # 3) Run prediction and write results
    with open(output_csv, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["filename", "Output"])
        for fname in filenames:
            full_path = os.path.join(video_dir, fname)
            try:
                preds = predict_video(full_path)           # returns List[str]
                writer.writerow([fname, ", ".join(preds)])
                print(f"✓ {fname} → {preds}")
            except Exception as e:
                writer.writerow([fname, f"ERROR: {e}"])
                print(f"✗ {fname} failed: {e}")

    print(f"\nDone — wrote {len(filenames)} rows to {output_csv}")

if __name__ == "__main__":
    run_inference(VIDEO_DIR, INPUT_FILE, OUTPUT_FILE, N)