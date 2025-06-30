import csv

annotations_file = "/research/lfcats/data/original/video_robustness/kinetics-dataset/k400/annotations/test.csv"
class_id_map_file = "/research/lfcats/projects/video_robustness/action_robustness/SlowFast/kinetics_400_labels.csv" 
output_map_file = "file_to_id.csv"

# Step 1: Load label → class_id mapping
label_to_id = {}
with open(class_id_map_file, "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        class_id = int(row[0])
        label = row[1].strip()
        label_to_id[label] = class_id

# Step 2: Process annotations to create filename → class_id mapping
with open(annotations_file, "r") as ann_f, open(output_map_file, "w", newline="") as out_f:
    reader = csv.DictReader(ann_f)
    writer = csv.writer(out_f)
    writer.writerow(["filename", "class_id"])

    for row in reader:
        label = row["label"]
        youtube_id = row["youtube_id"]
        start = int(row["time_start"])
        end = int(row["time_end"])
        split = row["split"]

        if split != "test":
            continue

        # Format the filename (same format as test video files)
        filename = f"{youtube_id}_{start:06d}_{end:06d}.mp4"

        # Look up class ID
        if label not in label_to_id:
            print(f"Missing label: {label}")
            continue
        class_id = label_to_id[label]

        writer.writerow([filename, class_id])