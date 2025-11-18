import pandas as pd

df = pd.read_csv("kaggle_file\\train_input\\input_2023_w01.csv")

condition = (df["player_to_predict"] == True)

count = df[condition].shape[0]

print("Rows matching conditions:", count)