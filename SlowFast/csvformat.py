import os
import csv

# Paths
VIDEO_DIR = "/research/lfcats/data/original/video_robustness/kinetics-dataset/k400/test"
OUTPUT_FILE = "/research/lfcats/projects/video_robustness/action_robustness/SlowFast/inputs.csv"

def convert_mp4_to_csv(video_dir: str, output_file: str):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for video_file in os.listdir(video_dir):
            if video_file.endswith('.mp4'):
                writer.writerow([video_file])

convert_mp4_to_csv(video_dir, output_file)