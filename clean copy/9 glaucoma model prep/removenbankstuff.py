import pandas as pd

negative_bank = pd.read_csv("negative_bank.csv")
negative_bank = negative_bank[negative_bank["Signal Strength"] >= 0.6]
negative_bank.to_csv("negative_bank_cleaned.csv", index=False)