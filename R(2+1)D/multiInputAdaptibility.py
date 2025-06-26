import pandas as pd
import newR21D
inputDF = pd.read_csv("inputs.csv")

for index, row in inputDF.iterrows():
    inputDF.loc[index,"Output"] = newR21D.R2PLUS1D(row["Input"])

inputDF.to_csv("outputs.csv", index = False)