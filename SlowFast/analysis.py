import csv

"""
Description:
    This script reads a CSV file containing video file names and prints each name.

Arguments:
    input (str): The path to the input CSV file.

Returns:
    None: The function prints the names of the video files.
"""
def analyze_csv(input: str):

    # Load class ID -> label mapping
    id_to_label = {}
    with open("utils/kinetics_400_labels.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_id = int(row[0])
            label = row[1].strip()
            id_to_label[class_id] = label

    # Load filename -> class ID mapping
    filename_to_id = {}
    with open("utils/file_to_id.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            filename = row[0].strip()
            class_id = int(row[1])
            filename_to_id[filename] = class_id

    # Load predictions from CSV
    predicted_labels = {}
    with open("outputs.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            filename = row[0].strip()
            predictions = [p.strip() for p in row[1].split(",")]
            predicted_labels[filename] = predictions
        
    # Compare predictions with actual labels
    total = 0
    top1_correct = 0
    top5_correct = 0
    missing = 0
    mistakes = []

    for filename, predictions in predicted_labels.items():

        # Skip if no predictions
        if filename not in filename_to_id:
            print(f"Warning: {filename} not found in file_to_id.csv")
            missing += 1
            continue

        file_id = filename_to_id[filename]
        correct_label = id_to_label[file_id]

        total += 1

        # Check if any of the predictions match the correct label
        if predictions[0] == correct_label:
            top1_correct += 1
 
        if correct_label in predictions:
            top5_correct += 1
 
        else:
            mistakes.append({
                "filename": filename,
                "correct_label": correct_label,
                "top1_prediction": predictions[0],
                "top5_predictions": predictions
            })
    
    # Output results
    print(f"\nTotal files processed: {total}")
    print(f"Top-1 accuracy: {top1_correct / total:.2%}")
    print(f"Top-5 accuracy: {top5_correct / total:.2%}")
    print(f"Missing files: {missing}")
    print(f"Mistakes: {len(mistakes)}")

    return mistakes


"""
Description:
    This function saves a list of misclassified video predictions to a CSV file.
    Each entry includes the filename, ground-truth label, top-1 prediction, and
    the full list of top-5 predictions.

Arguments:
    mistakes (List[Dict]): A list of dictionaries containing mistake details.
    filename (str): The path to the output CSV file. Defaults to 'mistakes.csv'.

Returns:
    None: The function writes data to a file but does not return a value.
"""
def save_mistakes(mistakes, filename="mistakes.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "correct_label", "top1_prediction", "top5_predictions"]
        )
        writer.writeheader()
        for m in mistakes:
            m_copy = m.copy()
            m_copy["top5_predictions"] = ", ".join(m["top5_predictions"])
            writer.writerow(m_copy)
                
if __name__ == "__main__":
    mistakes = analyze_csv("outputs.csv")

    # Save all mistakes to CSV
    save_mistakes(mistakes, filename="mistakes.csv")
    print("\nMistakes saved to mistakes.csv")
    