import csv
import informationFunctions as fn
from tree import *
from copy import deepcopy
import matplotlib.pyplot as plt

# Given a file path to a .names file, returns the possible classifications (categories)
# and the attribute names (featureSet)
def getDataNames(file):
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter='\n')
        rowNum = 0
        for row in reader:
            if rowNum == 0:
                r = row[0].replace(" ", "").split(',')
                categories = [int(value) for value in r]
                rowNum += 1
            elif rowNum == 1:
                featureSet = row[0].replace(" ", "").lower().split(',')
                rowNum += 1

    return categories, featureSet


# Returns an embedded list of all of the data, given a csv file.
def getData(file):
    with open(file) as csvfile:
        dataSet = []
        reader = csv.reader(csvfile, delimiter='\n')
        for row in reader:
            r = row[0].replace(" ", "").split(',')
            dataSet.append([int(value) for value in r])
    # print(csvfile.name)

    return dataSet

# Takes the dataSet and the list of featureSet and returns the feature that maximizes information gain
def argmaxInfo(dataSet, values):
    max = float("-inf")
    maxVal = ""
    for v in values:
        num = fn.infoGain(dataSet, values, v)
        if num > max:
            max = num
            maxVal = v

    return maxVal

# This uses a method to find the best partitioning of continuous variables as outlined in "On the Handling of Continuous-Valued
# Attributes in Decision Tree Generation" by Usama Fayyad and Keki Irani
def bestPartition(dataSet, featureSet):
    partitions = []
    for feature in featureSet:
        minEntropy = float("inf")
        values = fn.subsetFreq(dataSet, featureSet, feature)
        sorts = sorted(values)
        lenSorts = len(sorts)
        for n in range(1, lenSorts):
            partition1 = sorts[0:n]
            partition2 = sorts[n:lenSorts]
            set1 = [instance for instance in dataSet if instance[featureSet.index(feature)] in partition1]
            set2 = [instance for instance in dataSet if instance[featureSet.index(feature)] in partition2]
            entropy = ((len(partition1) / len(dataSet)) * fn.expectedInfo(set1)) + ((len(partition1) / len(dataSet)) * fn.expectedInfo(set2))
            if entropy < minEntropy:
                minEntropy = entropy
                partitions.append((partition1, partition2))
    return partitions



# This function recodes the data set such that all of the atrribute values take on Booleans
def preprocess(data, featureSet, partitions):

    for feature in featureSet:
        fIdx = featureSet.index(feature)
        for instance in data:
            if instance[fIdx] in partitions[fIdx][0]:
                instance[fIdx] = 0
            else:
                instance[fIdx] = 1


def pluralityValue(dataSet):
    max = 0
    major = 0
    categories = fn.getCategories(dataSet)
    for key in categories:
        if categories[key] > max:
            max = categories[key]
            major = key
    return major

# An implementation of the ID3 algorithm
# Here I introduce a splitCounter  variable. If we never remove an attribute from our set, the tree
# will continue to build indefinitely. This split variable allows us to control how many times we can split on a
# single attribute before cutting it from the set.

def learnTree(dataSet, featureSet, parentExamples, splitCount, cutoff):

    default = pluralityValue(parentExamples)
    categories = fn.getCategories(dataSet)
    if dataSet == []:
        return Node(default)
    elif len(categories) == 1:
        return Node(list(categories.keys())[0])
    elif featureSet == []:
        return Node(pluralityValue(dataSet))
    else:

        best = argmaxInfo(dataSet, featureSet)

        bestIdx = featureSet.index(best)
        splitCount[bestIdx] += 1
        bestVals = fn.subsetFreq(parentExamples, featureSet, best)

        tree = Node(best)
    # This allows us to remove an attribute if we've seen it the amount of times of the cutoff variable
        if splitCount[bestIdx] == cutoff:
            del featureSet[bestIdx]
            del splitCount[bestIdx]
            for val in bestVals:
                examples = [instance for instance in dataSet if instance[bestIdx] == val]
                subtree = learnTree(examples, featureSet, parentExamples, splitCount, cutoff)
                tree.split(val, subtree)
    # oTher
        else:
            for val in bestVals:
                examples = [instance for instance in dataSet if instance[bestIdx] == val]
                subtree = learnTree(examples, featureSet, parentExamples, splitCount, cutoff)
                tree.split(val, subtree)

        return tree



# Runs all instances of a test set and returns statistical information
def testTree(tree, testSet, featureSet, categories):
    testResults = []
    realClassifications = [line[len(line)-1] for line in testSet]
    total = len(realClassifications)
    matches = 0
    for instance in testSet:
        testResults.append(followTree(tree, instance, featureSet, categories))

    for idx in range(total):
        if realClassifications[idx] == testResults[idx]:
            matches += 1

    return matches/total


def main():
    namesFile = ""
    trainingFiles = []
    testingFile = ""
    pruningFile = ""
    print("This is decision tree learning program.")
    print("You can run the algorithm on the built in trees, or you can give the program custom training/testing sets.")
    print("Type '1' to run the balloons data.")
    print("Type '2' to run the breast cancer data.")
    print("Type '3' to run the handwriting data.")
    print("Or type '4' to specify a path to your own data. \n")
    choice = input("Enter your choice: ")
    print("\n")

    if choice=='1':
        print ("Showing learned trees for balloons data. \n")
        namesFile = "Learning/balloons/balloons.names"
        trainingFiles = ["Learning/balloons/yellow-small+adult-stretch.csv", "Learning/balloons/yellow-small.csv",
                         "Learning/balloons/adult-stretch.csv", "Learning/balloons/adult+stretch.csv"]
        categories, featureSet = getDataNames(namesFile)

        for file in trainingFiles:
            counter = []
            for feature in featureSet:
                counter.append(0)
            dataSet = getData(file)
            tree = learnTree(dataSet, featureSet, dataSet, counter, 5)
            print(file)
            print(tree.printTree())
            print("Tree depth is ", treeDepth(tree, 0))
            print(numNodes(tree,1), " nodes")
            print("\n")
        main()

    elif choice=='2':
        print("WDBC Data: ")
        print("Majority class accuracies: ")
        categories, featureS = getDataNames("Learning/wdbc/wdbc.names")
        testFeature = deepcopy(featureS)
        testFeature1 = deepcopy(featureS)
        trainData = getData("Learning/wdbc/wdbc-test.csv")
        testData = getData("Learning/wdbc/wdbc-test.csv")
        partitions = bestPartition(trainData, featureS)
        preprocess(trainData, featureS, partitions)
        preprocess(testData, featureS, partitions)
        splitCounter = []
        splitCounter1 = []
        for n in featureS:
            splitCounter.append(0)
            splitCounter1.append(0)
        for feature in featureS:
            tree = learnTree(trainData, [feature], trainData, splitCounter, 3)
            print(feature)
            print(testTree(tree, testData, testFeature, fn.getCategories(trainData))*100)
            print("\n")
        print("Full tree: ")
        tree = learnTree(trainData, featureS, trainData, splitCounter1, 3)
        print("Accuracy: ",testTree(tree, testData, testFeature1, fn.getCategories(trainData))*100)
        print("With a depth of: ", treeDepth(tree, 0))
        print(numNodes(tree, 1), " nodes")
        print(tree.printTree())
        print("\n")
        main()

    elif choice=='3':
        sets = ["Learning/digits/pen-10.csv", "Learning/digits/pen-20.csv", "Learning/digits/pen-30.csv",
                "Learning/digits/pen-40.csv", "Learning/digits/pen-50.csv", "Learning/digits/pen-60.csv"]
        xs = []
        for x in range(0,len(sets)): xs.append(x)
        categories, featureS = getDataNames("Learning/digits/pen.names")
        ## Pen2 and pen3
        pen2Features = featureS[:len(featureS)-9]
        pen3Features = []
        for n in range(len(featureS)-1):
            if n%2==0: pen3Features.append(featureS[n])
        pen2 = getData("Learning/digits/pen2.csv")
        preprocess(pen2, pen2Features, bestPartition(pen2, pen2Features))
        print("Loaded pen2...")
        pen3 = getData("Learning/digits/pen3.csv")
        preprocess(pen3, pen3Features, bestPartition(pen3, pen3Features))
        print("Loaded pen3...")

        print("Loading testing data...")
        testData = getData("Learning/digits/pen-test.csv")
        preprocess(testData, featureS, bestPartition(testData, featureS))

        print("Training pen2...")
        tree2 = learnTree(pen2, pen2Features[:], pen2, [0]*len(pen2Features), 20)
        print("Pen2 tests at", testTree(tree2, testData, pen2Features, categories)*100)
        print("With a depth of: ", treeDepth(tree2, 0))
        print(numNodes(tree2, 1), " nodes")

        print("Training pen3...")
        tree3 = learnTree(pen3, pen3Features[:], pen3, [0] * len(pen3Features), 20)
        print("Pen3 tests at", testTree(tree3, testData, pen3Features, categories)*100)
        print("With a depth of: ", treeDepth(tree3, 0))
        print(numNodes(tree3, 1), " nodes")



        totalData = []
        featCounter = []
        for feature in featureS:
            featCounter.append(0)
        print("Loading in training data...")
        results = []
        for fileName in sets:
            print(fileName)
            # Load in and process data
            data = getData(fileName)
            preprocess(data, featureS, bestPartition(data, featureS))
            print("Loaded training data...")
            #Aggregate all training examples
            totalData += data
            print("Training tree...")
            tree = learnTree(totalData, featureS[:], totalData, featCounter[:], 10)
            print("Testing...")
            results.append(testTree(tree, testData, featureS[:], categories)*100)
            print("NEXT")


        print("With a final depth of: ", treeDepth(tree, 0))
        print(numNodes(tree, 1), " nodes")
        plt.plot(xs, results)
        plt.plot([0,6,0,100])
        plt.xlabel("Number of training sets")
        plt.ylabel("Test accuracy")
        plt.show()

        main()

    elif choice == '4':
        namesFile = input("Please enter the path to the name info (as .names): ")
        trainingFile = input("Please enter the path to training files (as .csv): ")
        testingFile = input("Please enter the path to testing files (as .csv): ")

        categories, features = getDataNames(namesFile)
        trainData = getData(trainingFile)
        preprocess(trainData, features, bestPartition(trainData, features))
        testData = getData(testingFile)
        preprocess(testData, features, bestPartition(testData, features))

        tree = learnTree(trainData, features[:], trainData, [0]*len(features), 10)
        print("Trained data with an accuracy of: ", testTree())
        print("Depth: ", treeDepth(tree,0))
        print("Number of nodes: ", numNodes(tree,1))
        main()

    else:
        return





if __name__=='__main__':
    main()
