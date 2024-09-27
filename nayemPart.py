import pandas as pd

dataset1 = pd.read_csv('dataset1.csv')  
dataset2 = pd.read_csv('dataset2.csv') 
dataset3 = pd.read_csv('dataset3.csv')  

print("Dataset1 Demographics Information:")
print(dataset1.info())
print("\nDataset2 Screen Time Information:")
print(dataset2.info())
print("\nDataset3 Well-being Indicators Information:")
print(dataset3.info())

print("\nMissing values in Dataset1 Demographics:")
print(dataset1.isnull().sum())
print("\nMissing values in Dataset2 Screen Time:")
print(dataset2.isnull().sum())
print("\nMissing values in Dataset3 Well-being Indicators:")
print(dataset3.isnull().sum())

dataset1 = dataset1.dropna()
dataset2 = dataset2.dropna()
dataset3 = dataset3.dropna()

print("\nDemographic Distribution in Dataset1:")
print(dataset1.describe())
