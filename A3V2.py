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

# prints a dictionary containing mapping from class identifiers in the data to an index
def display_class_to_id(data):
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
    print("index: classLabel:::", classDictionary)
    return
    
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

# runs the naive bayes function on a training and test file
def naive_bayes(trainingFile, testFile):
    trainingData = open_file(trainingFile, 1)
    testData = open_file(testFile)

    display_class_to_id(trainingData) # display mapping of index to class label

    average = compute_averages(trainingData) # compute the averages for the training data
    stdev = compute_stdev(trainingData) # compute the standard deviation for the training data
    print_ave_std(trainingData, average, stdev) 
    class_probability = generate_class_probability(trainingData)



if __name__ == '__main__':
    #naive_bayes('pendigits_training.txt', 'pendigits_test.txt')
    naive_bayes('satellite_training.txt', 'satellite_test.txt')
    #naive_bayes('yeast_training.txt', 'yeast_test.txt')

    print("EOF")