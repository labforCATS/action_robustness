import os
import csv

# Paths
VIDEO_DIR = "/research/lfcats/data/original/video_robustness/kinetics-dataset/k400/test"
OUTPUT_FILE = "/research/lfcats/projects/video_robustness/action_robustness/SlowFast/inputs.csv"

"""
Description:
    This script scans a specified directory for MP4 video files and writes their names to a CSV file.

Arguments:
    video_dir (str): The directory containing the MP4 video files.
    output_file (str): The path to the output CSV file.

Returns:
    None: The function writes the names of the MP4 files to the specified CSV file.
"""
def convert_mp4_to_csv(video_dir: str, output_file: str):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for video_file in os.listdir(video_dir):
            if video_file.endswith('.mp4'):
                writer.writerow([video_file])

if __name__ == "__main__":
    convert_mp4_to_csv(VIDEO_DIR, OUTPUT_FILE)