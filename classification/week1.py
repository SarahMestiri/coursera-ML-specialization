from __future__ import division
import graphlab
import math
import string

products = graphlab.SFrame('amazon_baby.gl/')

print products[269]

#remove punctuation
def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

review_without_punctuation = products['review'].apply(remove_punctuation)
products['word_count'] = graphlab.text_analytics.count_words(review_without_punctuation)

print products[269]['word_count']

#consider rating = 3 as neutral
products = products[products['rating'] != 3]
len(products)

#assign rating = +1 for rating > 3 and rating = -1 for rating <3
products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1)

print products
train_data, test_data = products.random_split(.8, seed=1)
print len(train_data)
print len(test_data)

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                      target = 'sentiment',
                                                      features=['word_count'],
                                                      validation_set=None)
print sentiment_model

weights = sentiment_model.coefficients
print weights.column_names()

num_positive_weights = len(weights[weights['value']>=0])
num_negative_weights = len(weights[weights['value']<0])

print "Number of positive weights: %s " % num_positive_weights
print "Number of negative weights: %s " % num_negative_weights

sample_test_data = test_data[10:13]
print sample_test_data['rating']
print sample_test_data
print sample_test_data[0]['review']
print sample_test_data[1]['review']
scores = sentiment_model.predict(sample_test_data, output_type='margin')
print scores

def class_predict(scores):
    preds = []
    for score in scores:
      if score > 0:
        y = +1
      else:
        y=-1
      preds.append(y)
    return preds

print class_predict(scores)

print "Class predictions according to GraphLab Create:"
print sentiment_model.predict(sample_test_data)

def probability_predict(scores):
    prob = []
    for score in scores:
        p_positive = 1 / (1+ math.exp(-score))
        prob.append(p_positive)
    return prob

print probability_predict(scores)
print "Class predictions according to GraphLab Create:"
print sentiment_model.predict(sample_test_data, output_type='probability')

test_data['proba_pred'] = sentiment_model.predict(test_data, output_type='probability')
test_data
print test_data['name','proba_pred'].topk('proba_pred', k=20).print_rows(20)

print test_data['name','proba_pred'].topk('proba_pred', k=20, reverse= True).print_rows(20)


def get_classification_accuracy(model, data, true_labels):
    # First get the predictions
    ## YOUR CODE HERE
    predictions = model.predict(data)

    # Compute the number of correctly classified examples
    ## YOUR CODE HERE
    # compare 2 SArray, true = 1, false = 0
    num_correct = sum(predictions == true_labels)

    # Then compute accuracy by dividing num_correct by total number of examples
    ## YOUR CODE HERE
    accuracy = num_correct / len(data)

    return accuracy

print get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])

significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
      'work', 'product', 'money', 'would', 'return']

train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)

print train_data[0]['review']

print train_data[0]['word_count']
print train_data[0]['word_count_subset']

simple_model = graphlab.logistic_classifier.create(train_data,
                                                   target = 'sentiment',
                                                   features=['word_count_subset'],
                                                   validation_set=None)

print get_classification_accuracy(simple_model, test_data, test_data['sentiment'])

simple_weights = simple_model.coefficients
positive_significant_words = simple_weights[(simple_weights['value']>0) & (simple_weights['name']=='word_count_subset')]['index']
print simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21)
print len(positive_significant_words)
print positive_significant_words
weights.filter_by(positive_significant_words, 'index')

print get_classification_accuracy(sentiment_model, train_data, train_data['sentiment'])
print get_classification_accuracy(simple_model, train_data, train_data['sentiment'])

print 'accuracy on test data'
print get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
print get_classification_accuracy(simple_model, test_data, test_data['sentiment'])

num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == -1).sum()
print num_positive
print num_negative


print (test_data['sentiment'] == +1).sum()
print (test_data['sentiment'] == -1).sum()

print (test_data['sentiment'] == +1).sum()/len(test_data['sentiment'])