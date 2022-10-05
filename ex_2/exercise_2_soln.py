
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Exercise 1:

# Method one
df = pd.read_csv('auto.csv', sep=',', na_values='?')
df = df.dropna()
df['horsepower']

# Method two
df2 = pd.read_csv('auto.csv')
df2 = df2[df2.horsepower != '?']
df2['horsepower']


# Exercise 2:

# Don't plot against mpg or name
for i, name in enumerate(list(df.columns)):
	if name in ['mpg', 'name']:
		continue
	plt.subplot(2, 4, i)
	plt.plot(df[name], df['mpg'], '.', color='r', markersize=2)
	plt.ylabel('mpg')
	plt.xlabel(name)

plt.subplots_adjust(hspace=0.3, wspace=0.8)
plt.savefig("graph.png", dpi=200)
plt.show()


# Exercise 3:

# Plot the points
plt.plot(df['horsepower'], df['mpg'],'ro')
plt.ylabel('mpg'); plt.xlabel('horsepower')
plt.show()

# Fit the model
model = ols('mpg ~ horsepower', data=df).fit()
print(model.summary())

# Plot the line
horsepower_range = np.arange(min(df['horsepower']), max(df['horsepower']), 1)
hp_values = model.params.Intercept + model.params.horsepower * horsepower_range

plt.plot(df['horsepower'], df['mpg'],'ro')
plt.plot(horsepower_range, hp_values)
plt.ylabel('mpg'); plt.xlabel('horsepower'); plt.title(f'mpg regressed on Horsepower')
plt.savefig("horsepower.png", dpi=200)
plt.show()


# Exercise 4:
cmd1 = f'mpg ~ {" + ".join([f for f in list(df.columns) if f not in ["mpg", "name"]])}'
print(cmd1)
model = ols(cmd1, data=df).fit()
print(model.summary())
# Acceleration, horsepower and cylinders are not relevant for the model
cmd2 = f'mpg ~ {" + ".join([f for f in list(df.columns) if f not in ["mpg", "name", "acceleration", "horsepower", "cylinders"]])}'
print(cmd2)
model = ols(cmd2, data=df).fit()
print(model.summary())
# If we remove these, displacement now becomes irrelevant


# Exercise 5:
# Make it into a np.array for easy transformations!
df_array = np.array(df.iloc[:, : -1])

# log(X)
A = np.log(df_array)
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1])

model = ols(f'mpg ~ {" + ".join([f for f in list(df2.columns) if f not in ["mpg"]])}', data=df2).fit()
print(model.summary())
# R^2 = 0.89
# acceleration now have very low p value (before 0.415, now 0.003)


# sqrt(X)
A = np.sqrt(df_array)
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1])

model = ols(f'mpg ~ {" + ".join([f for f in list(df2.columns) if f not in ["mpg"]])}', data=df2).fit()
print(model.summary())
# R^2 = 0.866
# acceleration now have lower p value (before 0.415, now 0.165)
# horsepower now have much lower p value (before 0.220, now 0.002)


# X^2
A = np.square(df_array)
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1])

model = ols(f'mpg ~ {" + ".join([f for f in list(df2.columns) if f not in ["mpg"]])}', data=df2).fit()
print(model.summary())
# R^2 = 0.706
# all p-values low except horsepower which is 0.967


# 1/x
A = np.reciprocal(df_array)
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1])

model = ols(f'mpg ~ {" + ".join([f for f in list(df2.columns) if f not in ["mpg"]])}', data=df2).fit()
print(model.summary())
# R^2 = 0.852
# all p-values low except acceleration which is 0.825


# Extra: X^2 of only weight
A = df_array
A[:, 4] = np.square(A[:, 4])
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1])

model = ols(f'mpg ~ weight', data=df2).fit()
print(model.summary())
# R^2 = 0.65
# p-value for weight^2 is almost 0