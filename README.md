# QuantChallenge-2025
Python code that was used for the "Data" section of QuantChallenge 2025. 

The scripts employ different machine and deep learning models in order to predict the prices of two different assets based on market data. The objective of this part of the competition was to create the model with the highest accuracy in price estimation (through comparison of $R^2$ values with actual market data). The performance of our algorithms was evaluated through the application of standard statistical metrics. The team was awarded $1^{\text{st}}$ prize in the competition overall and this GitHub respotistory contains my contributions to the code. 

The scripts in this repository use the following machine learning models: 

* Light Gradient Boosting Model with a hyperparameter tuning pipeline (this model performed the best with the actual market data.
* Deep Learning Model that tests different Neural Network architectures, namely, basic feedforward, deep (including more layers), residual (skipping some connections to improve gradient descents), and wide (wide & deep architecture combining linear and deep). For the features themselves, feature engineering using statistical metrics has been employed to replace null values. To enhance accuracy, 5-fold cross-validation was used, reporting means and std of RMSEs across folds. Finally, to adjust the learning rate, cosine annealing was used, in addition to RobustScaler (to optimise outlier treatment) and L2 regularisation alongside Early Stopping. Despite varying the architecture across different trials, this model appeared to underperform, potentially due to its increased complexity leading to overfitting.
* Random Forest Regression that employs future selection based on the calculation of correlations with market data and therefore identifies feature imporance to isolate and use the most important features for price prediction. 
