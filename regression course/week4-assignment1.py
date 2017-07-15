import graphlab
import matplotlib.pyplot as plt
import numpy as np

def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_sframe[name] to be feature^power
            poly_sframe[name] =  feature.apply(lambda x: x**power)
    return poly_sframe

sales = graphlab.SFrame('kc_house_data.gl/')
sales = sales.sort(['sqft_living','price'])
l2_small_penalty = 1e-5

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price']
model15 = graphlab.linear_regression.create(poly15_data, target = 'price', l2_penalty=l2_small_penalty,  features = my_features, validation_set = None)
#plt.plot(poly15_data['power_1'],poly15_data['price'],'.', poly15_data['power_1'], model15.predict(poly15_data),'-')
#plt.title('plot for model 15')
#plt.show()
coef = model15.get("coefficients")
print coef
(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)

set_1_data = polynomial_sframe(set_1['sqft_living'], 15)
my_features = set_1_data.column_names()
set_1_data['price'] = set_1['price']
model_set1 =  graphlab.linear_regression.create(set_1_data, target = 'price', l2_penalty=1e5, features = my_features, validation_set = None)
coef_set1 = model_set1.get("coefficients")
#print 'coef set 1'
#print coef_set1
#plt.plot(set_1_data['power_1'],set_1_data['price'],'.',
 #        set_1_data['power_1'], model_set1.predict(set_1_data),'-')
#plt.title('plot for set 1')
#plt.show()

set_2_data = polynomial_sframe(set_2['sqft_living'], 15)
my_features = set_2_data.column_names()
set_2_data['price'] = set_2['price']
model_set2 =  graphlab.linear_regression.create(set_2_data, target = 'price', l2_penalty=1e5, features = my_features, validation_set = None)
coef_set2 = model_set2.get("coefficients")
#print 'coef set 2'
#print coef_set2
#plt.plot(set_2_data['power_1'],set_2_data['price'],'.',
    #     set_2_data['power_1'], model_set2.predict(set_2_data),'-')
#plt.title('plot for set 2')
#plt.show()

set_3_data = polynomial_sframe(set_3['sqft_living'], 15)
my_features = set_3_data.column_names()
set_3_data['price'] = set_3['price']
model_set3 =  graphlab.linear_regression.create(set_3_data, target = 'price',l2_penalty=1e5, features = my_features, validation_set = None)
coef_set3 = model_set3.get("coefficients")
#print 'coef set 3'
#print coef_set3
#plt.plot(set_3_data['power_1'],set_3_data['price'],'.',
       #  set_3_data['power_1'], model_set3.predict(set_3_data),'-')
#plt.title('plot for set 3')
#plt.show()

set_4_data = polynomial_sframe(set_4['sqft_living'], 15)
my_features = set_4_data.column_names()
set_4_data['price'] = set_4['price']
model_set4 =  graphlab.linear_regression.create(set_4_data, target = 'price', l2_penalty=1e5, features = my_features, validation_set = None)
coef_set4 = model_set4.get("coefficients")
#print 'coef set 4'
#print coef_set4
#plt.plot(set_4_data['power_1'],set_4_data['price'],'.',
       #  set_4_data['power_1'], model_set4.predict(set_4_data),'-')
#plt.title('plot for set 4')
#plt.show()

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)

#number of observations in the training data
n = len(train_valid_shuffled)
k=10 #10-fold cross validation
for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1)/k-1)
    print i, (start, end)

train_valid_shuffled[0:10] # rows 0 to 9

validation4 = train_valid_shuffled[5818:7758]
print int(round(validation4['price'].mean(), 0))

first_part = train_valid_shuffled[0:5818]
second_part = train_valid_shuffled[7758:n]
train4 = first_part.append(second_part)
print int(round(train4['price'].mean(), 0))
def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    avg_valid_error = []
    n = len(data)
    for i in xrange(k):
        #Compute starting and ending indices of segment i and call 'start' and 'end'
        start = (n * i) / k
        end = (n * (i + 1) / k - 1)
        #print i, (start, end)
        validation = data[start:end+1]
        first_part = data[0:start]
        second_part = data[end+1:n]
        train = first_part.append(second_part)
        model_set = graphlab.linear_regression.create(train, target=output_name, l2_penalty=l2_penalty, features=features_list,
                                                       validation_set=None)
        e = sum((model_set.predict(validation[features_list]) - validation[output_name]) ** 2)

        avg_valid_error.append(e)
    avg_valid_error = graphlab.SArray(avg_valid_error).mean()

    return avg_valid_error

poly_data = polynomial_sframe(train_valid_shuffled['sqft_living'],15)
my_features = poly_data.column_names()
poly_data['price'] = train_valid_shuffled['price']
y = []
for l2_penalty in(np.logspace(1, 7, num=13)):
    y.append(k_fold_cross_validation(10, l2_penalty, poly_data, 'price', my_features))
# Plot the l2_penalty values in the x axis and the cross-validation error in the y axis.
# Using plt.xscale('log') will make your plot more intuitive.
l2_search = np.logspace(1, 7, num=13)
plt.plot(l2_search, y)
plt.xscale('log')
plt.show()

lowest_l2 = l2_search[np.argmin(y)]
print lowest_l2

model_all = graphlab.linear_regression.create(poly_data, target='price', l2_penalty=lowest_l2, features=my_features, verbose = False,
                                              validation_set=None)

poly_test = polynomial_sframe(test['sqft_living'], 15)
poly_test['price'] = test['price']

predictions_test = model_all.predict(poly_test[my_features])
test_error = predictions_test - poly_test['price']
RSS_test = sum(test_error * test_error)

print RSS_test