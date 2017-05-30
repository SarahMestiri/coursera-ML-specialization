import graphlab
import numpy as np
import matplotlib.pyplot as plt

sales = graphlab.SFrame('kc_house_data_small.gl/')

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

def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    return (feature_matrix / norms,norms)

(train_and_validation, test) = sales.random_split(.8,seed=1)
(train,validation) = train_and_validation.random_split(.8,seed=1)

feature_list = ['bedrooms',
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
                'yr_renovated',
                'lat',
                'long',
                'sqft_living15',
                'sqft_lot15']
features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')

features_train, norms = normalize_features(features_train) # normalize training set features (columns)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms

print 'features associated to the first house in test set: '
print features_test[0]
print 'features associated to the 10th house in the training set: '
print features_train[9]

euclidean_distance = np.sqrt(np.sum((features_train[9] - features_test[0])**2))
print 'euclidean distance between 1st house and 10th house is: %s' % str(euclidean_distance)

#calculate distance between our query house and the 10 first houses in training set
distance_list = {}
for i in range(0,10):
    distance_list[i] = np.sqrt(np.sum((features_train[i] - features_test[0]) ** 2))
    print(i,distance_list[i])
print 'nearest house is: '
print min(distance_list.items(), key= lambda x : x[1])

#vectorize the distance between our query house and the 3 first houses in training set
for i in xrange(3):
    # compute the element - wise difference
    print features_train[i]-features_test[0]
    # should print 3 vectors of length 18

#verify the result - test
# verify that vectorization works
results = features_train[0:3] - features_test[0]
print results[0] - (features_train[0]-features_test[0])
# should print all 0's if results[0] == (features_train[0]-features_test[0])
print results[1] - (features_train[1]-features_test[0])
# should print all 0's if results[1] == (features_train[1]-features_test[0])
print results[2] - (features_train[2]-features_test[0])
# should print all 0's if results[2] == (features_train[2]-features_test[0])

diff = features_train - features_test[0]
print diff[-1].sum()
print np.sum(diff**2, axis=1)[15] # take sum of squares across each row, and print the 16th sum
print np.sum(diff[15]**2) # print the sum of squares for the 16th row -- should be same as above

distances = np.sqrt(np.sum(diff**2,axis=1))
print 'euclidean distance between all training set houses and our query house'
print distances[100]

#compute euclidean distance between all training set rows and our query house
def compute_distances(features_matrix,query_vector):
   diff = features_matrix - query_vector
   distances = np.sqrt(np.sum(diff**2,axis=1))
   return distances

query_house = features_test[2]
distances = compute_distances(features_train,query_house)
print np.argsort(distances)
print 'predicted value based on 1-nearest neighbor regression %d'% output_train[np.argsort(distances)[0]]

# get k-nearest neighbors
def fetchKNN(k,features_matrix,query_vector):
    distances = compute_distances(features_matrix,query_vector)
    return np.argsort(distances)[:k]
print fetchKNN(4,features_train,query_house)

#define a function that computes prediction by averaging k nearest neghbor outputs
def predictByAvgKNN(k,features_matrix,output,query_vector):
    k_neighbors = fetchKNN(k,features_matrix,query_vector)
    avg_value = np.mean(output[k_neighbors])
    return avg_value

print 'predicted value by averaging k nearest neighbor outputs %d' % predictByAvgKNN(4,features_train,output_train,query_house)

def multiple_predictions(k,features_matrix,output,query_matrix):
    predicted_values = []
    for i in range((query_matrix.shape)[0]):
        avg_value = predictByAvgKNN(k,features_matrix,output,query_matrix[i])
        predicted_values.append(avg_value)
    return predicted_values

predictions =  multiple_predictions(10,features_train,output_train,features_test[:10])
print predictions

print 'index of the house in this query set that has the lowest predicted value: %d, its predicted value: %d' % (np.argsort(predictions)[0], min(predictions))


def get_RSS(predictions, output):
    # Then compute the residuals/errors
    residual = output - predictions

    # Then square and add them up
    residual_squared = residual * residual

    RSS = residual_squared.sum()

    return (RSS)

rss_all = []
for k in range(1,16):
   predictions =  multiple_predictions(k,features_train,output_train,features_valid)
   #compute RSS
   RSS = get_RSS(predictions,output_valid)
   rss_all.append(RSS)
print rss_all
print rss_all.index(min(rss_all))

kvals = range(1, 16)
plt.plot(kvals, rss_all,'bo-')
plt.show()

predictions = multiple_predictions(7, features_train, output_train, features_test)
RSS = get_RSS(predictions,output_test)
print RSS








