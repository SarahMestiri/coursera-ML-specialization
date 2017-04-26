import graphlab
import numpy as np
import matplotlib.pyplot as plt
tmp = graphlab.SArray([1., 2., 3.])
tmp_cubed = tmp.apply(lambda x: x**3)
print tmp
print tmp_cubed

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

poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price']

model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)


#plt.plot(poly1_data['power_1'],poly1_data['price'],'.', poly1_data['power_1'], model1.predict(poly1_data),'-')
#plt.title('plot for model 1')
#plt.show()

poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price']
model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)
#plt.plot(poly2_data['power_1'],poly2_data['price'],'.', poly2_data['power_1'], model2.predict(poly2_data),'-')
#plt.title('plot for model 2')
#plt.show()

poly3_data = polynomial_sframe(sales['sqft_living'], 3)
my_features = poly3_data.column_names() # get the name of the features
poly3_data['price'] = sales['price'] # add price to the data since it's the target
model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = my_features, validation_set = None)
#plt.plot(poly3_data['power_1'],poly3_data['price'],'.',poly3_data['power_1'], model3.predict(poly3_data),'-')
#plt.show()

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target
model15 = graphlab.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None)
coef = model15.get("coefficients")
#print coef
#plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
 #       poly15_data['power_1'], model15.predict(poly15_data),'-')
plt.show()

sub1,sub2 = sales.random_split(.5,seed=0)
set_1,set_2 = sub1.random_split(.5,seed=0)
set_3,set_4 = sub2.random_split(.5,seed=0)

set_1_data = polynomial_sframe(set_1['sqft_living'], 15)
my_features = set_1_data.column_names()
set_1_data['price'] = set_1['price']
model_set1 =  graphlab.linear_regression.create(set_1_data, target = 'price', features = my_features, validation_set = None)
coef_set1 = model_set1.get("coefficients")
plt.plot(set_1_data['power_1'],set_1_data['price'],'.',
         set_1_data['power_1'], model_set1.predict(set_1_data),'-')
plt.title('plot for set 1')
plt.show()

set_2_data = polynomial_sframe(set_2['sqft_living'], 15)
my_features = set_2_data.column_names()
set_2_data['price'] = set_2['price']
model_set2 =  graphlab.linear_regression.create(set_2_data, target = 'price', features = my_features, validation_set = None)
coef_set2 = model_set2.get("coefficients")
plt.plot(set_2_data['power_1'],set_2_data['price'],'.',
         set_2_data['power_1'], model_set2.predict(set_2_data),'-')
plt.title('plot for set 2')
plt.show()

set_3_data = polynomial_sframe(set_3['sqft_living'], 15)
my_features = set_3_data.column_names()
set_3_data['price'] = set_3['price']
model_set3 =  graphlab.linear_regression.create(set_3_data, target = 'price', features = my_features, validation_set = None)
coef_set3 = model_set3.get("coefficients")
plt.plot(set_3_data['power_1'],set_3_data['price'],'.',
         set_3_data['power_1'], model_set3.predict(set_3_data),'-')
plt.title('plot for set 3')
plt.show()

set_4_data = polynomial_sframe(set_4['sqft_living'], 15)
my_features = set_4_data.column_names()
set_4_data['price'] = set_4['price']
model_set4 =  graphlab.linear_regression.create(set_4_data, target = 'price', features = my_features, validation_set = None)
coef_set4 = model_set4.get("coefficients")
plt.plot(set_4_data['power_1'],set_4_data['price'],'.',
         set_4_data['power_1'], model_set4.predict(set_4_data),'-')
plt.title('plot for set 4')
plt.show()

training_and_validation,testing = sales.random_split(.9,seed=1)
training,validation = training_and_validation.random_split(.5,seed=1)
for degree in range(1,15+1):
    poly_data = polynomial_sframe(training['sqft_living'],degree)
    vali_data = polynomial_sframe(validation['sqft_living'], degree)
    test_data = polynomial_sframe(testing['sqft_living'], degree)
    my_features = poly_data.column_names()
    poly_data['price'] = training['price']
    poly_model = graphlab.linear_regression.create(poly_data, target = 'price', features = my_features, validation_set = None, verbose = False)
    predictions = poly_model.predict(vali_data)
    predictions_test = poly_model.predict(test_data)
    validation_errors = predictions - validation['price']
    test_errors = predictions_test - testing['price']
    RSS = sum(validation_errors * validation_errors)
    RSS_test = sum(test_errors * test_errors)
    print "degree : " + str(degree) + ", RSS : " + str(RSS) + ", RSS_test : " + str(
        RSS_test) + ", Training loss : " + str(poly_model.get('training_loss'))


