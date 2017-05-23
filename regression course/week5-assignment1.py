import graphlab
from math import log, sqrt
import numpy as np

sales = graphlab.SFrame('kc_house_data.gl/')
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']

# In the dataset, 'floors' was defined with type string,
# so we'll convert them to float, before creating a new feature.
sales['floors'] = sales['floors'].astype(float)
sales['floors_square'] = sales['floors']*sales['floors']
all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']
model_all = graphlab.linear_regression.create(sales, target='price', features=all_features,
                                              validation_set=None,
                                              l2_penalty=0., l1_penalty=1e10)
print model_all.get("coefficients")
print('number of nonzeros = %d' % (model_all.coefficients['value']).nnz())

(training_and_validation, testing) = sales.random_split(.9,seed=1) # initial train/test split
(training, validation) = training_and_validation.random_split(0.5, seed=1) # split training into train and validate
l1_penalty = np.logspace(1, 7, num=13)
rss_array = []
for x in l1_penalty:
    model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None,
                                              l2_penalty=0., l1_penalty=x, verbose=False)
    #get predictions
    predictions = model.predict(validation)
    #get output
    output = validation['price']
    #compute error
    error = output - predictions
    #compute residuals square
    residuals_squared = error * error
    #compute RSS
    RSS = residuals_squared.sum()
    print("l1_penalty: %s, RSS: $%.6f" % (x, RSS))
    rss_array.append((x,RSS))
# sort the RSS array by rss error
print np.sort(rss_array)
l1_penalty_best = np.sort(rss_array)[0][0]
print ("best l1_penalty: %s, RSS: $%.6f" %(rss_array[0][0],rss_array[0][1]))
model_lowest_rss = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None,
                                              l2_penalty=0., l1_penalty=l1_penalty_best, verbose=False)
print('number of nonzeros with the best l1_penalty = %d' % (model_lowest_rss.coefficients['value']).nnz())
max_nonzeros = 7
l1_penalty_values = np.logspace(8, 10, num=20)
list_nnz = []
for l1_penalty in l1_penalty_values:
    #create a model to train taining data using l1_penalty
    model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None,
                                              l2_penalty=0., l1_penalty=l1_penalty, verbose=False)
    #get number of non zero weights
    nnz = model.coefficients['value'].nnz()
    print ("l1_penalty: %s, number of nonzeroes: %d" % (l1_penalty,nnz))
    #append the result to the list of nnz weights
    list_nnz.append((l1_penalty,nnz))
# find the smallest l1_penalty that has fewer non-zeros than max_nonzeros
l1_penalty_min = min(t[0] for t in list_nnz if t[1] > max_nonzeros)
# find the  largest l1_penalty that has more non-zeros than max_nonzeros
l1_penalty_max = max(t[0] for t in list_nnz if t[1] < max_nonzeros)

print(l1_penalty_max,l1_penalty_min)

l1_penalty_values = np.linspace(l1_penalty_min,l1_penalty_max,20)
rss_array = []
for l1_penalty in l1_penalty_values:
    model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None,
                                              l2_penalty=0., l1_penalty=l1_penalty, verbose=False)
    #get predictions
    predictions = model.predict(validation)
    #get error
    error = validation['price'] - predictions
    #compute square of residuals
    residuals_squared = error * error
    #compte RSS on validation set
    RSS = residuals_squared.sum()
    nnz = model.coefficients['value'].nnz()
    rss_array.append((l1_penalty,RSS, nnz))
#initialize best RSS to first row RSS
best_rss = rss_array[0][1]
#initialize best L1_penalty to first row L1_penalty
bestL1 = rss_array[0][0]
for i in range(len(rss_array)):
    if (rss_array[i][2] == max_nonzeros) and (rss_array[i][1] < best_rss):
        best_rss = rss_array[i][1]
        bestL1 = rss_array[i][0]
print (best_rss, bestL1)
best_model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None,
                                              l2_penalty=0., l1_penalty=bestL1, verbose=False)
nonzero_weigths = best_model.coefficients[best_model['coefficients']['value']>0]
print nonzero_weigths






