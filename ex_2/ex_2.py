"""
Problem
The overall idea of this exercise is to predict the fuel consumption of cars (measured in miles-per
gallon, mpg) for various cars based on a linear regression model. The dataset is available at
the course website auto.csv
What to do?

• Download the auto.csv from the course website and load it into python. Use the pandas.read_csv
function for importing the dataset. Be aware, there are some missing values in the dataset,
indicated by ?. You have to remove those lines and then make sure the corresponding
columns are casted to a numerical type.

• Inspect the data. Plot the relationships between the different variables and mpg. Use for
example the matplotlib.pyplot scatter plot. Do you already suspect what features might
be helpful to regress the consumption? Save the graph.

• Perform a linear regression using the OLS function from the statsmodels package. Use
horsepower as feature and regress the value mpg. It is a good idea to look up the statsmod-
els documentation on OLS, to under- stand how to use it. Further, plot the results including
your regression line.

• Now extend the model using all features. How would you determine which features are
important and which aren’t? Try to find a good selection of features for your model.

• Can you improve your regression performance by trying different transformations of the
variables, such as log x, √x, 1
x , x2 and so on. Why are some transformations better?
"""