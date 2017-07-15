import graphlab
import numpy as np
from math import log
sales = graphlab.SFrame('kc_house_data.gl/')
train_data,test_data = sales.random_split(.8,seed=0)
#new variables in order to apply multiple regression
train_data['bedrooms_squared'] = train_data['bedrooms'] * train_data['bedrooms']
train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms']
train_data['log_sqft_living'] = [log(train_data['sqft_living'][i]) for i in range(len(train_data))]
train_data['lat_plus_long'] = train_data['lat'] + train_data['long']

test_data['bedrooms_squared'] = test_data['bedrooms'] * test_data['bedrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms'] * test_data['bathrooms']
#a = float(test_data['sqft_living'])
test_data['log_sqft_living'] = [log(test_data['sqft_living'][i]) for i in range(len(test_data))]
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

mean_bedrooms_squared = test_data['bedrooms_squared'].mean()
print round(mean_bedrooms_squared,2)
mean_bed_bath_rooms = test_data['bed_bath_rooms'].mean()
print round(mean_bed_bath_rooms,2)
mean_log_sqft_living = test_data['log_sqft_living'].mean()
print round(mean_log_sqft_living,2)
mean_lat_plus_long = test_data['lat_plus_long'].mean()
print round(mean_lat_plus_long,2)

model_1 = graphlab.linear_regression.create(train_data, target = 'price', features=['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long'], validation_set= None )
model_2 = graphlab.linear_regression.create(train_data, target = 'price', features=['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms'], validation_set= None )
model_3 = graphlab.linear_regression.create(train_data, target = 'price', features=['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'log_sqft_living', 'lat_plus_long'], validation_set= None )

#print model_1.get('coefficients')
#print model_2.get('coefficients')

print model_1.evaluate(test_data)
print model_2.evaluate(test_data)
print model_3.evaluate(test_data)

def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predictions = model.predict(data)
    # Then compute the residuals/errors
    residuals = outcome - predictions
    # Then square and add them up
    RSS = np.square(residuals).sum()
    return(RSS)

print get_residual_sum_of_squares(model_1, test_data, test_data['price'])
print get_residual_sum_of_squares(model_2, test_data, test_data['price'])
print get_residual_sum_of_squares(model_3, test_data, test_data['price'])