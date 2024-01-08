In this project, we first visually and briefly examined the data before preprocessing them. For example, we saw the first 10 and last rows of the data, and the columns and contents of each one. We then reviewed the number of rows and columns, a summary of each attribute, etc. Other things that may be of interest to us include how many empty values each column has, or what the values of each column are and how many times each value has been repeated. We printed a few graphs to further examine the data and detect anomalies. These graphs show us the relationship between the columns better.

Then, we randomly divided the data into two sets, train and test. We then worked on the train data and set aside the test data for evaluating the final model.

To better understand the relationship between the variables, we calculated the correlation matrix. We examined the features with the highest correlation to the label in more detail and, by plotting scatter plots for each one, visually observed this relationship and the strength of it. The income_median feature had the highest correlation to the target variable. The maximum price of houses was 500,000, which is likely, and based on this graph, it can be said that the prices of houses more than this amount are also 500,000! This can be a flaw.

In the previous step, we realized that the bedrooms_total variable was the only variable with missing values. Therefore, we need to think about how to fill these values with a suitable value in the preprocessing stage. Considering that this variable followed a normal distribution, we used the mean strategy to fill in the missing values.

We also added other derived columns from the initial columns to our dataframe. These columns can help us to better understand the issue. The columns we added are household_per_rooms, population_per_household, and bedrooms_per_room.

To avoid bias in the model, it is better to standardize the data. Accordingly, we standardized all numerical data using zscore.

The last column, which included the labels, was the only column with non-numeric values. We can convert it to a numerical value using different encoders, or map each value of this column to a number. This makes it possible to use this dataframe for use in future models.

Each model is an algorithm with a specific input, so it is important that our input dataframe follows those features. We used onehotencoder, which adds a column to the data for each value of this column, and sets the corresponding column to 1 and the rest to 0, for the value that the row has in this variable.

These steps are going to be done for the test data as well, so it is better to put the steps in sequence as a pipeline so that we can preprocess any new data using this pipeline.

Now it is time to try the data on different models and select the final model and result. The models that we will run include:

LinearRegression-1
DecisonTreeRegressor-2
Randomforestregressor-3

We applied the data to each of these models, and to see if these models are suitable, we used the error-squared-mean metric to calculate the mean difference between the predicted value and the actual value.

To get a better intuition, we printed the predicted value and the actual value for the first four rows in the output for the first algorithm, so that we have a better understanding of the performance of the algorithm.

The performance of the decision tree was such that the mean error was zero! This output is not necessarily a good output! It can be the result of overfitting the model.

To prevent overfitting, we used validation cross, which, with each execution of the algorithms, divides different parts of the data into train and test.

Therefore, we executed each algorithm 10 times and each time with a separate test data and considered a criterion for selecting the better model, including:

error_squared_mean_neg
The results of this step for tree decision are as follows:


mean_error = -2.7755575615628914e-17

This means that the decision tree model perfectly predicted the target variable for the training data. However, this is not a good sign, as it indicates that the model is overfitting the training data.

To evaluate the performance of the model on unseen data, we need to use a validation set. We will use the validation cross method to do this.

With the execution of validation cross for the linear regression algorithm and comparing the mean scores in these two algorithms:

In this step, it is observed that the error value in the regression algorithm is approximately 68,000, which is less than this amount for the decision tree algorithm. Therefore, the regression algorithm is the better algorithm.

For the random forest algorithm, the mean scores in 10 runs are 49,888.327139651155, which is clearly better than the previous two algorithms! So we choose this algorithm as the final model! But with what parameters?

But we don't know in advance what parameters are suitable for our work. For example, n_estimators is the number of trees based on which we make a decision. As you know, this algorithm uses several decision tree algorithms. By default, we have 100=n_estimators.

There is another feature called max_features that says for each split (in a tree) when we want to go from one branch to another, how many features should we consider in each split.

To compare different parameters, we use gridsearch. Scikit-learn has a class called GridSearchCV that gives us the ability to give it the numbers we want for the two parameters n_estimator and max_features, and then it runs with these values itself.

We give the desired values to GridSearchCv and fit the model! We print the best combination of hyperparameters and the best estimator (ie the model with the best combination of hyperparameters).

Finally, we determine the final model based on the best estimator and prepare the test data for prediction.

Preparing the test data is not difficult! We use pipeline_full that we wrote before. And then we do the prediction using the final model on the test data.

Next, we calculate the mean squared error (MSE) for the predictions of the final model and then calculate the root mean squared error (RMSE) and print it. RMSE indicates the standard deviation of prediction errors relative to actual values.

Result:

The final model (random forest model with parameters 'n_estimators=6', 'features_max=30') works on average with a difference of $47,705.24 from the actual values. This means that the predictions of the final model are on average $47,705.24 away from the actual values. This amount can be used as a measure of the performance of the model.
