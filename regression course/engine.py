import graphlab
import numpy as np
sales = graphlab.SFrame('kc_house_data.gl/')
train_data,test_data = sales.random_split(.8,seed=0)
#simple linear  function based on closed-form solution
def simple_linear_regression(input_feature, output):
    # compute the product of the output and the input_feature
    product =  sum([input_feature[i] * output[i] for i in range(len(output))])
    # compute the squared value of the input_feature and its sum
    square_x = sum(np.square(input_feature))
    # use the formula for the slope
    slope = (product  - (sum(input_feature)*sum(output))/len(output))/(square_x-(sum(input_feature)**2/len(output)))
    # use the formula for the intercept
    intercept = (sum(output) - slope * sum(input_feature))/len(output)
    return(intercept, slope)

input_feature = train_data['sqft_living']
output = train_data['price']

print simple_linear_regression(input_feature,output)

def get_regression_predictions(input_feature, intercept, slope):
    predicted_output = list()
    output = ([slope * input_feature[i] + intercept for i in range(len(input_feature))])
    predicted_output.append(output)
    return(predicted_output)

print get_regression_predictions([2650], -47116.076574939987, 281.9588385676974)


def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    RSS = 0.0
    for i in range(len(input_feature)):
     predicted_output = slope * input_feature[i] + intercept
     error = output[i] - predicted_output
     RSS += pow(error,2)
    return(RSS)
print get_residual_sum_of_squares(input_feature, output, -47116.076574939987, 281.9588385676974)
def inverse_regression_predictions(output, intercept, slope):
    estimated_input = (output - intercept)/slope
    return(estimated_input)

print inverse_regression_predictions(800000, -47116.076574939987, 281.9588385676974 )

input_bedroom_feature = train_data['bedrooms']

print simple_linear_regression(input_bedroom_feature,output)

intercept_bedroom, slope_bedroom = simple_linear_regression(input_bedroom_feature,output)

input_test_living_feature = test_data['sqft_living']
input_test_bedrooms_feature = test_data['bedrooms']
output_test = test_data['price']

rss_living_feature = get_residual_sum_of_squares(input_test_living_feature,output_test, -47116.076574939987, 281.9588385676974)
rss_bedrooms_feature = get_residual_sum_of_squares(input_test_bedrooms_feature, output_test, intercept_bedroom, slope_bedroom)
print rss_living_feature
print rss_bedrooms_feature
