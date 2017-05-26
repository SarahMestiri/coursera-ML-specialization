import graphlab
import numpy as np
import math

sales = graphlab.SFrame('kc_house_data.gl/')
sales['floors'] = sales['floors'].astype(int)

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

def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    return (feature_matrix / norms,norms)

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)

#normalize features
simple_feature_matrix, norms = normalize_features(simple_feature_matrix)

#intitialize random weights
weights = np.array([1., 4., 1.])

#predict output
prediction = predict_outcome(simple_feature_matrix,weights)
def compute_ro(feature_matrix,output,prediction,weights):

  ro = [0 for i in range((feature_matrix.shape)[1])]
  for i in range(len(weights)):
    ro[i] = sum(feature_matrix[:,i] * (output - prediction + weights[i] * feature_matrix[:,i] ))
  return ro

ro = compute_ro(simple_feature_matrix,output,prediction,weights)
print ro
l1_penalty_list = [1.4e8,1.64e8,1.73e8,1.9e8,2.3e8]

def inl1_range(value,l1_penalty):

    return ((value >= -l1_penalty/2) and (value <= l1_penalty/2))

for i in l1_penalty_list:
    print("l1_penalty %s"%i)
    print (inl1_range(ro[1],i),inl1_range(ro[2],i))


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    #weights = np.array(weights)
    # compute prediction
    prediction = predict_outcome(feature_matrix,weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = np.dot(feature_matrix[:, i], (output - prediction + weights[i] * feature_matrix[:, i]))


    if i == 0:  # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = (ro_i + l1_penalty/2.)
    elif ro_i > l1_penalty / 2.:
        new_weight_i = (ro_i - l1_penalty/2.)
    else:
        new_weight_i = 0.

    return new_weight_i

print lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],[2./math.sqrt(13),3./math.sqrt(10)]]),
                                   np.array([1., 1.]), np.array([1., 4.]), 0.1)
def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    weights = np.array(initial_weights)
    #initialize change coordinate
    change = np.array(initial_weights) * 0.0
    converged = False
    D = feature_matrix.shape[1]
    while not converged:
      for i in range(D):
        #memorize old weights of feature i
         old_weights_i = weights[i]
        #calculate the new weight of feature i
         new_weight =  lasso_coordinate_descent_step(i,feature_matrix,output,weights,l1_penalty)
         #print '  -> old weight: ' + str(weights[i]) + ', new weight: ' + str(new_weight)
         #print '  -> abs change (new - old): ' + str(change[i])
         #print '  >> old weights: ', weights
        #compute change in coordinate
         change[i]= np.abs(new_weight - old_weights_i)

        #assign the new weight
         weights[i] = new_weight

      max_change = max(change)
      if (max_change < tolerance):
         converged = True
    return weights


initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix) # normalize features

weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
prediction = predict_outcome(normalized_simple_feature_matrix,weights)

print weights
error = output - prediction
residuals_squared = error * error
RSS = residuals_squared.sum()
print ' RSS for l1_penalty 1e7, simple model: $%f' % RSS

train_data,test_data = sales.random_split(.8,seed=0)

all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated']

(all_features_matrix,output) = get_numpy_data(train_data, all_features, my_output)

normalized_all_features_matrix, all_norms = normalize_features(all_features_matrix)


initial_weights = np.zeros(14)
weights1e7 = lasso_cyclical_coordinate_descent(normalized_all_features_matrix, output,
                                            initial_weights, 1e7, 1.)

print 'weights1e7 are: '
print(zip(['constant'] + all_features, weights1e7))

weights1e8 = lasso_cyclical_coordinate_descent(normalized_all_features_matrix, output,
                                            initial_weights, 1e8, 1.)

print(zip(['constant'] + all_features, weights1e8))

weights1e4 = lasso_cyclical_coordinate_descent(normalized_all_features_matrix, output,
                                            initial_weights, 1e4, 5e5)
print(zip(['constant'] + all_features, weights1e4))

#normalized version of weights l1penalty = 1e7
weights1e7_normalized = weights1e7 / all_norms

#normalized version of weights1e8
weights1e8_normalized = weights1e8 / all_norms

#normalized version of weights1e4
weights1e4_normalized = weights1e4 / all_norms

print weights1e7_normalized[3]

(test_feature_matrix, test_output) = get_numpy_data(test_data, all_features, 'price')

predictions = predict_outcome(test_feature_matrix,weights1e7_normalized)
residuals = test_output - predictions
residuals_squared = residuals * residuals
RSS_1e7 = residuals_squared.sum()

print 'RSS for l1_penalty 1e7: $%.6f' % RSS_1e7

predictions = predict_outcome(test_feature_matrix,weights1e8_normalized)
residuals = test_output - predictions
residuals_squared = residuals * residuals
RSS_1e8 = residuals_squared.sum()
print 'RSS for l1_penalty 1e8: $%.6f' % RSS_1e8
predictions = predict_outcome(test_feature_matrix,weights1e4_normalized)
residuals = test_output - predictions
residuals_squared = residuals * residuals
RSS_1e4 = residuals_squared.sum()
print 'RSS for l1_penalty 1e4: $%.6f' % RSS_1e4



