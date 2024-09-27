import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

mergedData = pd.merge(pd.merge(dataset1, dataset2, on='ID'), dataset3, on='ID')

mergedData['computerTotal'] = mergedData['C_we'] + mergedData['C_wk']
mergedData['gamesTotal'] = mergedData['G_we'] + mergedData['G_wk']
mergedData['smartPhoneTotal'] = mergedData['S_we'] + mergedData['S_wk']
mergedData['tvTotal'] = mergedData['T_we'] + mergedData['T_wk']

correlationMatrix = mergedData.corr()

screenTimeVar = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk', 'computerTotal', 'smartPhoneTotal']
wellBeingVar = ['Optm', 'Relx', 'Conf']
correlations = correlationMatrix.loc[screenTimeVar, wellBeingVar]
print("\nCorrelations between screen time and well-being indicators:")
print(correlations)

plt.figure(figsize=(14, 10))
sns.heatmap(correlations, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap Between Screen Time and Well-being Indicators')
plt.xlabel('Well-being Indicators')
plt.ylabel('Screen Time Variables')
plt.show()

firstGroup = mergedData[mergedData['S_we'] > mergedData['S_we'].median()]['Optm']
secondGroup = mergedData[mergedData['S_we'] <= mergedData['S_we'].median()]['Optm']

tStats, pValue = stats.ttest_ind(firstGroup, secondGroup)
print('Hypothesis Test - Smartphone Use (Weekends) and Optimism:\nT-statistic: {tStats}, Pvalue: {pValue}')

firstGroup = mergedData[mergedData['G_we'] > mergedData['G_we'].median()]['Relx']
secondGroup = mergedData[mergedData['G_we'] <= mergedData['G_we'].median()]['Relx']

tStats, pValue = stats.ttest_ind(firstGroup, secondGroup)
print('Hypothesis Test of Video Games Weekends and Relaxation:\nT-statistic: {tStats}, Pvalue: {pValue}')
