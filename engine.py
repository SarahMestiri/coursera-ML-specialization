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

#print simple_linear_regression(input_feature,output)

def get_regression_predictions(input_feature, intercept, slope):
    predicted_output = slope * input_feature + intercept
    return(predicted_output)

print get_regression_predictions(2650, -47116.076574939987, 281.9588385676974)

#not completed
def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    predicted_output = get_regression_predictions(input_feature, intercept, slope)
    RSS = sum(output - predicted_output)
    return(RSS)
#print get_residual_sum_of_squares(train_data,test_data,-47116.076574939987,281.9588385676974)