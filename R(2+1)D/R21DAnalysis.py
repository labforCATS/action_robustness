import pandas as pd
df = pd.read_csv("outputs.csv")

top1 = 0.0
top5 = 0.0
total = 0.0

for index, row in df.iterrows():
    currentLabel = row["label"]
    currentPrdictions = row["Predicted Output"]
    total += 1
    if currentLabel in currentPrdictions.split()[0]:
        top1 += 1
    if currentLabel in currentPrdictions:
        top5 += 1
    if (currentLabel not in currentPrdictions.split()[0]) and (currentLabel not in currentPrdictions):
        print("Error Video: ", row["youtube_id"])
        print(row["label"])
        print(row["Predicted Output"])
        print("----------------------------------------")
    
print("Top 1%: ", (top1/total)*100)
print("Top 5%: ", (top5/total)*100)
print(total)