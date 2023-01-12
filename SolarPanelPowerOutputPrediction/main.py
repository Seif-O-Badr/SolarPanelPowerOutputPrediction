import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# drop first column, since it is just numbering
df = pd.read_csv('AswanData_weatherdata.csv', index_col=0)
df_original = df.copy()

# data discovery
plt.plot(df_original['Date'], df_original['AvgTemp'], color='green', label='AvgTemp')
plt.plot(df_original['Date'], df_original['AverageDew'], color='black', label='AvgDew')
plt.plot(df_original['Date'], df_original['Humidity'], color='red', label='Humidity')
plt.plot(df_original['Date'], df_original['Wind'], color='blue', label='Wind')
plt.xticks(('2021-04-30', '2021-07-30', '2021-10-30', '2022-01-30', '2022-04-01'),
           ('2021-04-30', '2021-07-30', '2021-10-30', '2022-01-30', '2022-04-01'))
plt.xlabel('Date')
plt.title('Correlation According to Date')
plt.legend(loc='upper left')
plt.show()


plt.scatter(df_original['AvgTemp'], df_original['Solar(PV)'])
plt.title('Solar(PV) According to AvgTemp')
plt.xlabel('Average Temperature')
plt.ylabel('Solar(PV)')
plt.show()

# data printing
print(df)

# data discovery
print(df.describe())

# data format
print(df.info())
print('============================================================================')
# no null or missing values, there is no need to drop rows, so there is no
# need to fill in missing data or make dummies
# but, the 'date' column will not be useful for indicating a change in solar(PV),
# since the date itself is irrelevant

df = df.drop('Date', axis=1)
sns.pairplot(data=df, height=1.25)
plt.show()

# convert into numpy dataframe
X = df.values
y = df['Solar(PV)'].values
# dropping Solar(PV) from X
X = np.delete(X, 5, axis=1)
# splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# variance threshold
vThreshold = VarianceThreshold()
X_variance = vThreshold.fit(X_train, y_train)
print(X_variance.get_support())
print('Variance Threshold : \n', X_variance.get_support())

# multiple linear regression
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
y_predict = reg.predict(X_test)
print('Predicted Values : \n', y_predict)

# SSE calculation
sse = np.sum((y_predict - y_test) ** 2)
print('SSE : ', sse)

# regression coefficients
print('Coefficients: ', reg.coef_)

# regression intercept
print('Intercept: ', reg.intercept_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plot for residual error

# setting plot style
plt.style.use('fivethirtyeight')

# residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="blue", s=10, label='Train data')

# residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="red", s=10, label='Test data')

# plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

# plotting legend
plt.legend(loc='upper right')

# plot title
plt.title("Residual errors plot")

# method call for showing the plot
plt.show()

# Random Forest and Square Root of MSE, default number of n_estimators
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))
