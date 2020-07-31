# Written for Assignment 4 by Austen Hsiao, 985647212
import numpy as np
import pandas as pd
import statistics as stat
import math

# opens file by file name. sort=1 will sort the file by the value in the last column (class id)
def open_file(file, sort=0):
    file = pd.read_csv(file, sep='[ ]+', header=None, engine='python').to_numpy()
    if sort == 1:
        file = file[np.argsort(file[:, -1])]

        sortedByClass = []
        row = 0
        while row < len(file):
            dataForCurrentClass = []
            currentClass = file[row][-1]
            while file[row][-1] == currentClass:
                dataForCurrentClass.append(file[row])
                row += 1
                if row >= len(file):
                    break
            sortedByClass.append(dataForCurrentClass)
        file = np.array(sortedByClass, dtype=object)
    return file

# returns a dictionary containing mapping from class identifiers in the data to an index
def class_to_id(data):
    classDictionary = {}
    dictionaryi = 0
    for classNum in range(len(data)):
        row = 0
        while row < len(data[classNum]):
            currentClass = data[classNum][row][-1]
            classDictionary[dictionaryi] = currentClass
            while data[classNum][row][-1] == currentClass:
                row += 1
                if row >= len(data[classNum]):
                 break
            dictionaryi += 1
    return classDictionary
    
# returns a matrix containing the averages of each feature. eg. matrix[2][5] --> average for 6th feature of class 2
def compute_averages(dataMatrix):
    average = []
    for classNum in range(len(dataMatrix)):
        local_average = []
        for i in (np.transpose(dataMatrix[classNum]))[:-1]:
            mean = stat.fmean(i)
            local_average.append(mean)
        average.append(local_average)
    return average

# returns a matrix containing the standard deviations of each feature. eg. matrix[2][5] --> standard deviation for 6th feature of class 2
def compute_stdev(dataMatrix):
    std = []
    for classNum in range(len(dataMatrix)):
        local_std = []
        for i in (np.transpose(dataMatrix[classNum]))[:-1]:
            calc_std = stat.stdev(i)
            if calc_std < 0.01:
                calc_std = 0.01
            local_std.append(calc_std)
        std.append(local_std)
    return std

# prints data related to class, attribute, mean, and standard deivation
def print_ave_std(data, averageMatrix, stdevMatrix):
    for classNum in range(len(data)):
        for featureNum in range(len(averageMatrix[0])):
            print("Class %d, attribute %d, mean = %.2f, std = %.2f" % (classNum, (featureNum+1), averageMatrix[classNum][featureNum], stdevMatrix[classNum][featureNum]))

# generates a list that holds the probabilities of each class occuring. eg. list[2] --> P(class = 2)
def generate_class_probability(data):
    classProbability = []
    total = 0
    for classes in data:
        total += len(classes)
        classProbability.append(len(classes))
    return list(map(lambda x: x/total, classProbability))

# returns the probability density function to be used in the Naive Bayes calculations
def pdf(average, standarddeviation, inputx):
    firstPhrase = 1/(math.sqrt(2*math.pi) * standarddeviation)
    secondPhraseNumerator = math.pow((inputx - average), 2)
    secondPhraseDenominator = 2 * math.pow(standarddeviation, 2)
    secondPhrase = math.exp(-secondPhraseNumerator/ secondPhraseDenominator)
    return firstPhrase * secondPhrase

#helper function that returns a key based on the value-- we are guaranteed that the classes are unique
def getKey(val, dictionary):
    for key, value in dictionary:
      if val == value:
         return key
      return "key doesn't exist"

def run_test_data(testData, classProbability, average, standardDev, dictionary):
    hit = 0 # whether or not at the very end, there is a hit (successful prediction)
    objectID = 1 # used below to identify lines in the test data
    for test in testData: # out the outer loop, we process the test set time by time
        pdfs = [] # pdfs for the line of data
        for classNum in range(len(average)): # length of average is the number of classes we have
            probabilitiesCurrentClass = []
            for featureNum in range(len(average[0])): # length of average[0] is the number of features we have
                ave = average[classNum][featureNum] # pulling the average out of the matrix
                std = standardDev[classNum][featureNum] # pulling the stdev out of the matrix
                x = test[featureNum] # input
                p = pdf(ave, std, x) # pdf() returns the pdf for a given average, stdev, and input
                probabilitiesCurrentClass.append(p) 
            pdfs.append(probabilitiesCurrentClass)  # add each line of probabilities (pdfs, rather) to pdfs. With the way I've set up the for-loops,
                                                    # pdfs[0] will contain the list of probabilities for each feature given class 0
        
        product = [np.prod(prob)*classprob for prob, classprob in zip(pdfs,classProbability)] # for each list, calculate the product of all pdfs
        predictedClassIndex = product.index(max(product)) # still need to look up what class label the actual index is associated with
        duplicateIndices = [index for index, value in zip(enumerate(product), product) if value == max(product)] # iterate through the product list 
                                                                                                                 #(with the final calculations for each class) to check for duplicates

        # check for successes. Note that we have to convert the classindices to the real class label using the dictionary. I converted
        # the true class label and dictionary values to ints otherwise we might have errors trying to equate floats. The implication is that we have to change behavior if 
        # the classes cannot be represented by ints.
        if len(duplicateIndices) == 1 and int(dictionary.get(predictedClassIndex)) == int(test[-1]):
            accuracy = 1
        elif len(duplicateIndices) == 1 and int(dictionary.get(predictedClassIndex)) != int(test[-1]):
            accuracy = 0
        elif len(duplicateIndices) > 1 and GetKey(int(test[-1]), dictionary) in duplicateIndices:
            accuracy = 1/len(duplicateIndices)
        elif len(duplicateIndices) > 1 and GetKey(int(test[-1]), dictionary) not in duplicateIndices:
            accuracy = 0

        # if the accuracy is anything bigger than 0, we have a match and we'll add it to the hits.
        if accuracy > 0:
            hit += accuracy

        print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (objectID, dictionary.get(predictedClassIndex), max(product), int(test[-1]), accuracy))
        objectID += 1

    print("\nclassification_accuracy=%6.4f" % (hit/len(testData)))

# runs the naive bayes function on a training and test file
def naive_bayes(trainingFile, testFile):
    trainingData = open_file(trainingFile, 1) # open and parse trainingData
    testData = open_file(testFile) # open and parse testData

    dictionary = class_to_id(trainingData) # dictionary contains map from index to real label
    print("Index: classLabel::::", dictionary)

    average = compute_averages(trainingData) # compute the averages for the training data
    stdev = compute_stdev(trainingData) # compute the standard deviation for the training data
    print_ave_std(trainingData, average, stdev) # prints out a string for each class's attribute that includes the mean and standard deviation
    class_probability = generate_class_probability(trainingData) # create an array that has the probabilities for each class classProbability[0] = P(class = 0)

    run_test_data(testData, class_probability, average, stdev, dictionary) # Run and print accuracy

if __name__ == '__main__':
    ## Uncomment the one to run. or load a new one in with a new line

    #naive_bayes('pendigits_training.txt', 'pendigits_test.txt')
    #naive_bayes('satellite_training.txt', 'satellite_test.txt')
    naive_bayes('yeast_training.txt', 'yeast_test.txt')
    print("End")