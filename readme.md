ARIMA Cross-Validation

1.2.2) I would do more or less the same as in the first part, besides defining the train and test differently, 
only I would provide actual time series indices
I would try to fit the ARIMA model with a Kalman filter instead of maximal likelihood, 
which might deal better with the missing values.