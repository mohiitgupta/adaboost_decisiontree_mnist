from random import shuffle
import pickle
import csv

def shuffle_data():
    with open('winequality-white.csv', 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ';')
        # print type(csvreader)
        data = []
        for i, row in enumerate(csvreader):
            if i != 0:
                row = [float(i) for i in row]
                data.append(row)
    shuffle(data)
    with open('shuffled_data_wine', 'wb') as fp:
        pickle.dump(data, fp)

"""
The function divide folds is for dividing the data into k folds which is later
used for cross validation.
"""
def divide_folds(k, data):
    if k < 0:
        return [data]

    fold_size = len(data)/k
    folds = [data[x:x+fold_size] for x in xrange(0, len(data), fold_size)]
    if len(folds) == 5:
        folds[3] += folds[4]
        del folds[4]
    return folds

def split_folds_train_test(test_index, folds):
    test_fold = folds[test_index]
    train_fold = []
    for x in range(len(folds)):
        if x != test_index:
            train_fold += folds[x]

    return train_fold, test_fold

def split_train_validation(train_data):
    validation_len = len(train_data)/5
    validation = train_data[:validation_len]
    train = train_data[validation_len:]
    # print validation[validation_len-1]
    # print train[0]
    return train, validation

def find_normalization_params(train_data):
    minValue = [x for x in train_data[0]]
    maxValue = [x for x in train_data[0]]
    for example in train_data:
        for i in range(len(example)):
            if example[i]>maxValue[i]:
                # print "max ", example[i], " ", maxValue[i]
                # break
                maxValue[i]=example[i]
            if example[i]<minValue[i]:
                minValue[i]=example[i]

    return minValue, maxValue

def min_max_norm(value, minValue, maxValue, scale, isInt):
    value = (value-minValue)/(maxValue-minValue)*scale
    if value < 0:
        value = 0
    if value > scale:
        value = scale
    if (isInt):
     return int(value)
    return value

def normalize(data, minValue, maxValue, ignored_feature, scale, isInt):
    features, labels = [],[]
    for example in data:
        normalized_example = []
        for j in range(len(minValue)-1):
            if j not in ignored_feature:
                normalized_example.append(min_max_norm(example[j], minValue[j], maxValue[j], scale, isInt))
        labels.append(int(example[len(minValue)-1]))
        features.append(normalized_example)

    return features, labels