import pandas as pd

localdfPath = "C:/Users/Team 818/Documents/GitHub/senior-research-neural-network/emg_data_classified.csv"
localdf = pd.read_csv(localdfPath)

z = localdf["classification"]
print(z)