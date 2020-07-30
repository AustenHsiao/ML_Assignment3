# Written for Assignment 4 by Austen Hsiao, 985647212
import numpy as np
import pandas as pd
import statistics as stat
import math

# the assignment says to write a function for naive bayes, so Im writing everything in a monolithic function

def naive_bayes(trainingFile, testFile):
    try:
        # parse the files into sorted matrices. (sorted by class-- the last column)
        trainingData = pd.read_csv(trainingFile, sep='[ ]+', header=None, engine='python').to_numpy()
        trainingData = trainingData[np.argsort(trainingData[:, -1])]

        testData = pd.read_csv(testFile, sep='[ ]+', header=None, engine='python').to_numpy()
        #testData = testData[np.argsort(testData[:, -1])] # do i want to sort this?
        
        # sort the data in such a way that each element will be a matrix associated with 1 class
        # classDictionary provides information as to which index maps to which class
        classSortedTrainingData = []
        classDictionary = {}
        dictionaryi = 0
        row = 0
        while row < len(trainingData):
            dataForCurrentClass = []
            currentClass = trainingData[row][-1]
            classDictionary[dictionaryi] = currentClass
            while trainingData[row][-1] == currentClass:
                dataForCurrentClass.append(trainingData[row])
                row += 1
                if row >= len(trainingData):
                    break
            classSortedTrainingData.append(dataForCurrentClass)
            dictionaryi += 1
        trainingData = np.array(classSortedTrainingData, dtype=object)

        average = []
        standardDev = []

        # if we take the transpose, each row contains data for a single feature (attribute) for the class label.
        # The average and stdev are stored in average and standardDev, respectively. WLOG, average[3][4] -> average of class 3, attribute 5
        # Note that this doesn't mean that our class is necessarily 3, just that it is the 3rd class label.
        print("_____TRAINING SET DATA_____\n")
        for classLabel in range(len(trainingData)):
            attribute = 1
            local_average = []
            local_std = []
            # the mean and standard deviation for each row will be outputted except for the last row since it will consist of the class label
            for featureDataLine in np.transpose(trainingData[classLabel])[:-1]:
                mean = stat.fmean(featureDataLine)
                stdev = stat.stdev(featureDataLine)
                if stdev < 0.01: # make sure that variance is above the minimum
                    stdev = 0.01

                local_average.append(mean)
                local_std.append(stdev)
                print("Class %d, attribute %d, mean = %.2f, std = %.2f" % (classLabel, attribute, mean, stdev))
                attribute += 1
            average.append(local_average)
            standardDev.append(local_std)
        print("\nClass dictionary (index: ActualClassName):: ", classDictionary)
        ######################## calculate the P(class) for each class label ########
        
        #classProbability[0] --> P(class index 0). See classDictionary to map indices to actual labels
        classProbability = []
        total = 0
        for classes in trainingData:
            total += len(classes)
            classProbability.append(len(classes))
        classProbability = list(map(lambda x: x/total, classProbability))

        ######################## Using test set below ###############################
        print("\n\n_____TEST SET DATA_____\n\n")
        objectID = 1 
        accuracyHit = 0
        # At the end of these ugly loops, probability will contain the probabilities for the current line of testData. eg probability[1][3] --> P(attribute4 | class 1)
        for test in testData:
            probability = []
            for classNum in range(len(classDictionary)):
                probabilityClass = []
                for featureNum in range(len(average[0])):
                    ave = average[classNum][featureNum]
                    std = standardDev[classNum][featureNum]                    
                    p = (1 / (math.sqrt(2*math.pi) * std)) * math.exp( -math.pow((test[featureNum] - ave), 2) / (2 * math.pow(std, 2)))
                    #print(p, "input:", test[featureNum], "average:", ave, "std:", std) #for debugging
                    probabilityClass.append(p)
                probability.append(probabilityClass)

            predictedClass = []
            for classNum in range(len(classDictionary)):
                predictedClass.append(classProbability[classNum] * math.prod(probability[classNum]))
            prediction = predictedClass.index(max(predictedClass)) # gives an index
            probability = predictedClass[prediction] # gives the probability
            indexMatch = [index for index, value in enumerate(predictedClass) if value == probability]
            duplicates = predictedClass.count(probability)
            if duplicates <= 1 and prediction == test[-1]:
                acc = 1
            elif duplicates <= 1 and prediction != test[-1]:
                acc = 0
            elif duplicates > 1 and prediction in indexMatch:
                acc = 1/len(indexMatch)
            elif duplicates > 1 and prediction not in indexMatch:
                acc = 0
            if acc > 0:
                accuracyHit += 1
            print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (objectID, prediction, probability, test[-1], acc))
            objectID += 1

        print(len(testData))
        #print("\nclassification_accuracy=%6.4f" % (accuracyHit/len(testData)))
    except:
        print("Something went wrong. Exiting...")
        return 

if __name__ == '__main__':
    naive_bayes('pendigits_training.txt', 'pendigits_test.txt')
    #naive_bayes('satellite_training.txt', 'satellite_test.txt')
    #naive_bayes('yeast_training.txt', 'yeast_test.txt')