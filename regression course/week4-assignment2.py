import graphlab
import numpy as np
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


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    if(feature_is_constant):
        derivative = 2 * np.dot(errors, feature)
    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    else:
        derivative = 2 * np.dot(errors, feature) + 2 * l2_penalty * weight

    return derivative


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty,
                                      max_iterations=100):
    print 'Starting gradient descent with l2_penalty = ' + str(l2_penalty)

    weights = np.array(initial_weights)  # make sure it's a numpy array
    iteration = 0  # iteration counter
    print_frequency = 1  # for adjusting frequency of debugging output

    #
    while not iteration > max_iterations:
        iteration += 1  # increment iteration counter

        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_outcome(feature_matrix,weights)
        # compute the errors as predictions - output
        errors = predictions - output

        for i in xrange(len(weights)):  # loop over each weight

         if(i==0):
           derivative = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, True)
         else:
          derivative = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, False)

         # subtract the step size times the derivative from the current weight
         weights[i] = weights[i] - (step_size * derivative)
    print 'Done with gradient descent at iteration ', iteration
    print 'Learned weights = ', str(weights)

    return weights

simple_features = ['sqft_living']
my_output = 'price'
train_data,test_data = sales.random_split(.8,seed=0)
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 0.0,
                                      max_iterations)
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 1e11,
                                      max_iterations)
import matplotlib.pyplot as plt
plt.plot(simple_feature_matrix,output,'k.',
         simple_feature_matrix,predict_outcome(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_outcome(simple_feature_matrix, simple_weights_high_penalty),'r-')
#plt.show()
predictions = predict_outcome(simple_test_feature_matrix,initial_weights)

test_errors = predictions - test_output
RSS = np.square(test_errors).sum()
print 'RSS is: '
print RSS

predictions_no_regu = predict_outcome(simple_test_feature_matrix,simple_weights_0_penalty)
test_errors = predictions_no_regu - test_output
RSS = np.square(test_errors).sum()
print 'RSS with no regulations ' + str(RSS)

predictions_regu = predict_outcome(simple_test_feature_matrix,simple_weights_high_penalty)
test_errors = predictions_regu - test_output
RSS = np.square(test_errors).sum()
print 'RSS with regulations ' + str(RSS)

model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors.
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

initial_weights = np.array([0.0,0.0,0.0])
step_size = 1e-12
max_iterations = 1000

l2_penalty = 0.0
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty,
                                      max_iterations)
l2_penalty=1e11
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty,
                                      max_iterations)

predictions = predict_outcome(test_feature_matrix,initial_weights)
test_errors = test_output - predictions
RSS = np.square(test_errors).sum()
print 'RSS initial weights ' + str(RSS)

predictions_no_regu = predict_outcome(test_feature_matrix,multiple_weights_0_penalty)
print 'first house price using no regulations: ' + str(predictions_no_regu[0])
print 'RSS on predicted price ' + str(abs(test_output[0]-predictions_no_regu[0]))
test_errors = predictions_no_regu - test_output
RSS = np.square(test_errors).sum()
print 'RSS with no regulations ' + str(RSS)

predictions_regu = predict_outcome(test_feature_matrix,multiple_weights_high_penalty)
print 'first house price using regulations: ' + str(predictions_regu[0])
print 'real price is: ' + str(test_output[0])
print 'RSS on predicted price ' + str(abs(test_output[0]-predictions_regu[0]))

test_errors = predictions_regu - test_output
RSS = np.square(test_errors).sum()
print 'RSS with regulations ' + str(RSS)






