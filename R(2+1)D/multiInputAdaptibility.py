import pandas as pd
import newR21D
inputDF = pd.read_csv("inputs.csv")

for index, row in inputDF.iterrows():
    time_start = row["time_start"]
    time_end = row["time_end"]
    vidSuffix = "_"+f"{time_start:06d}"+"_"+f"{time_end:06d}"
    vidName = "/research/lfcats/data/original/video_robustness/kinetics-dataset/k400/val/"+row["youtube_id"]+vidSuffix+".mp4"
    inputDF.loc[index,"Predicted Output"] = newR21D.R2PLUS1D(vidName)
    print(vidName)

inputDF.to_csv("outputs.csv", index = False)