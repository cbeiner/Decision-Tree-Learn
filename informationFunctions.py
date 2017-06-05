import math
# This file contains functions that deal with calculating entropy and splitting the data set into subsets

# Gives a dictionary with each of the categories as well as the number
# of occurences of each category in the data set
def getCategories(dataSet):
    count = {}
    for entry in dataSet:
        category = entry[len(entry) - 1]
        if category in count:
            count[category] += 1
        else:
            count[category] = 1
    return count


# Gives a dictionary that has the possible values of a feature
# as well as the number of occurences of that value in the data set
def subsetFreq(dataSet, featureSet, desiredAttribute):
    subset = {}
    attribute = featureSet.index(desiredAttribute)
    for entry in dataSet:
        x = entry[attribute]
        if x in subset:
            subset[x] += 1
        else:
            subset[x] = 1
    return subset

# Returns a subset of the data in which the specified feature equals the specified value
def getSubset(dataSet, featureSet, attribute, value):
    return [entry for entry in dataSet if entry[featureSet.index(attribute)] == value]


# Returns the entropy of the data set
def expectedInfo(dataSet):
    cats = getCategories(dataSet)
    values = [((cats[category] / len(dataSet)) * math.log(cats[category] / len(dataSet), 2)) for category in cats]
    return -sum(values)


# Returns the entropy of a specified subset of the data
def featInfo(dataSet, featureSet, attribute):
    featVals = subsetFreq(dataSet, featureSet, attribute)
    summ = 0
    for val in featVals:
        subset = [instance for instance in dataSet if instance[featureSet.index(attribute)]==val]
        summ += ((len(subset) / len(dataSet)) * expectedInfo(subset))
    return summ


# Difference of expected entropy and feature entropy
def infoGain(dataSet, featureSet, attribute):
    return expectedInfo(dataSet)-featInfo(dataSet, featureSet, attribute)
