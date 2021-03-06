import graphlab
import numpy as np
from math import sqrt
sales = graphlab.SFrame('kc_house_data.gl/')

def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the 'features' list into the SFrame 'features_sframe'
    features_sframe = data_sframe[features]
    # this will convert the features_sframe into a numpy matrix with GraphLab Create >= 1.7!!
    features_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the target to the variable 'output_sarray'
    output_sarray = data_sframe[output]
    # this will convert the SArray into a numpy array:
    output_array = output_sarray.to_numpy() # GraphLab Create>= 1.7!!
    return(features_matrix, output_array)

def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix,weights)
    return(predictions)
def feature_derivative(errors, feature):
    derivative = 2 * np.dot(errors,feature)
    return(derivative)


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions = predict_outcome(feature_matrix,weights)
        # compute the errors as predictions - output:
        errors = predictions - output
        gradient_sum_squares = 0  # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
         # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
         # compute the derivative for weight[i]:
         derivative =feature_derivative(errors, feature_matrix[:,i])
         # add the squared derivative to the gradient magnitude
         gradient_sum_squares += derivative * derivative
         # update the weight based on step size and derivative:
         weights[i] = weights[i] - (step_size * derivative)

         gradient_magnitude = sqrt(gradient_sum_squares)
         if gradient_magnitude < tolerance:
            converged = True
    return (weights)

train_data,test_data = sales.random_split(.8,seed=0)

simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)
print 'Simple weights are: '
print simple_weights
print round(simple_weights[1], 1)

#build test simple feature matrix and test output using test data
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features,
     my_output)
simple_predicted_outcome = predict_outcome(test_simple_feature_matrix,simple_weights)
print 'predicted price of 1st house: '
print round(simple_predicted_outcome[0])

test_errors = simple_predicted_outcome - test_output
RSS = np.square(test_errors).sum()
print 'RSS is: '
print RSS

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features,my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9
(feature_matrix, test_output) = get_numpy_data(train_data, model_features,my_output)
multiple_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
print 'Multiple weights are: '
print multiple_weights
print round(multiple_weights[1], 1)

multiple_predictions = predict_outcome(feature_matrix,multiple_weights)
print 'price of 1st house model 2: '
print round(multiple_predictions[0])

print 'real price'
print test_data['price'][0]
multiple_test_errors = multiple_predictions - test_output
RSSm = sum(multiple_test_errors * multiple_test_errors)
print RSSm

print RSS < RSSm
