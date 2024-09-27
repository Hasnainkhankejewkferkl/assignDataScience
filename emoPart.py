import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')
mergedData = pd.merge(pd.merge(dataset1, dataset2, on='ID'), dataset3, on='ID')
mergedData['computerTotal'] = mergedData['C_we'] + mergedData['C_wk']
mergedData['gamesTotal'] = mergedData['G_we'] + mergedData['G_wk']
mergedData['smartPhoneTotal'] = mergedData['S_we'] + mergedData['S_wk']
mergedData['tvTotal'] = mergedData['T_we'] + mergedData['T_wk']
mergedData['C_diff'] = mergedData['C_we'] - mergedData['C_wk']
mergedData['G_diff'] = mergedData['G_we'] - mergedData['G_wk']
mergedData['S_diff'] = mergedData['S_we'] - mergedData['S_wk']
mergedData['T_diff'] = mergedData['T_we'] - mergedData['T_wk']
mergedData['genderSWe'] = mergedData['gender'] * mergedData['S_we']
mergedData['minoritySWe'] = mergedData['minority'] * mergedData['S_we']
X = mergedData[['C_we', 'G_we', 'S_we', 'T_we', 'computerTotal', 'gamesTotal', 'smartPhoneTotal', 'tvTotal', 
                 'C_diff', 'G_diff', 'S_diff', 'T_diff', 'genderSWe', 'minoritySWe']]
y = mergedData['Optm']  
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
linearPipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

rfPipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

linearPipeline.fit(xTrain, yTrain)
yPredLR = linearPipeline.predict(xTest)
mseLR = mean_squared_error(yTest, yPredLR)
print(f'Mean Squared Error of the Linear Regression model: {mseLR}')

rfPipeline.fit(xTrain, yTrain)
yPredRF = rfPipeline.predict(xTest)
mseRF = mean_squared_error(yTest, yPredRF)
print(f'Mean Squared Error of the Random Forest model: {mseRF}')

cvScoresLR = cross_val_score(linearPipeline, X, y, cv=5, scoring='neg_mean_squared_error')
cvScoresRF = cross_val_score(rfPipeline, X, y, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-validated MSE Linear Regression: {-cvScoresLR.mean()}')
print(f'Cross-validated MSE Random Forest: {-cvScoresRF.mean()}')

plt.figure(figsize=(10, 6))
plt.scatter(yTest, yPredLR)
plt.xlabel('Actual Well-being Scores Optimism')
plt.ylabel('Predicted Well-being Scores (Linear Regression)')
plt.title('Actual vs Predicted Well-being Scores (Linear Regression)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(yTest, yPredRF)
plt.xlabel('Actual Well-being Scores Optimism')
plt.ylabel('Predicted Well-being Scores (Random Forest)')
plt.title('Actual vs Predicted Well-being Scores (Random Forest)')
plt.grid(True)
plt.show()
