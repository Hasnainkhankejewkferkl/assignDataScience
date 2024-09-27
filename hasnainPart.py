import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset2 = pd.read_csv('dataset2.csv')  
dataset3 = pd.read_csv('dataset3.csv')  

print("\nScreen Time Summary in Dataset2:")
print(dataset2[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].describe())

print("\nWell-being Indicators Summary in Dataset3:")
print(dataset3.describe())

plt.figure(figsize=(12, 8))
sns.histplot(dataset2['C_we'], kde=True, label='Computer in Weekend', color='blue')
sns.histplot(dataset2['C_wk'], kde=True, label='Computer in Weekday', color='lightblue')
sns.histplot(dataset2['G_we'], kde=True, label='Video Games in Weekend', color='green')
sns.histplot(dataset2['G_wk'], kde=True, label='Video Games in Weekday', color='lightgreen')
sns.histplot(dataset2['S_we'], kde=True, label='Smartphone in Weekend', color='orange')
sns.histplot(dataset2['S_wk'], kde=True, label='Smartphone in Weekday', color='blue')
sns.histplot(dataset2['T_we'], kde=True, label='TV in Weekend', color='red')
sns.histplot(dataset2['T_wk'], kde=True, label='TV in Weekday', color='lightcoral')
plt.legend()
plt.title('Screen Time Distribution for All Devices')
plt.xlabel('Screen Time Hours')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(dataset3['Optm'], kde=True, label='Optimism', color='blue')
sns.histplot(dataset3['Relx'], kde=True, label='Relaxation', color='green')
plt.legend()
plt.title('Well-being Scores Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()
