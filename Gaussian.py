import math
import csv

class GaussianNaiveBayes:
    def __init__(self):
        pass

    # handle data
    def loadcsv(self,data,header=False):
        lines =  csv.reader(open(data, "rb"))
        dataset = list(lines)
        if header:
            dataset = dataset[1:]
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
        return dataset

    def separateByClass(self,data):
        classes = {}
        for i in range(len(data)):
            vector = data[i]
            if (vector[-1] not in classes): # assuming class labels are in the last column
                classes[vector[-1]] = []
            classes[vector[-1]].append(vector)
        return classes

    def mean(self,data):
        return sum(data)/float(len(data))

    def variance(self,data):
        avg = self.mean(data)
        variance = sum([pow(x-avg,2) for x in data])/float(len(data)-1)
        return variance

    def standardDeviation(self,data):
        avg = self.mean(data)
        variance = sum([pow(x-avg,2) for x in data])/float(len(data)-1)
        return math.sqrt(variance)

    def summarizeDataset(self,data):
        data = [sublist[1:] for sublist in data]
        summaries = [(self.mean(attribute),self.standardDeviation(attribute)) for attribute in zip(*data)]
        summaries_with_variance = [(self.mean(attribute),self.variance(attribute)) for attribute in zip(*data)]
        del summaries_with_variance[-1]
        del summaries[-1]   # remove class labels summaries
        return summaries

    def summarizeByClass(self,data):
        separated = self.separateByClass(data)
        summaries = {}
        for classvalue, vectors in separated.iteritems():
            summaries[classvalue] = self.summarizeDataset(vectors)
        return summaries

    # Calculate Priors
    def calculatePriors(self,data):
        priors = {}
        separated = self.separateByClass(data)
        for key,vectors in separated.iteritems():
            if key not in priors.keys():
                priors[key] = []
            priors[key].append(len(separated[key])/float(len(data)))
        return priors

    def predict(self,data,summaryPerClass,priors):
        _training_features = [sublist[0:-1] for sublist in data]
        _training_predictions = {}
        for i in range(1,len(_training_features) + 1):
            _training_predictions[_training_features[i-1][0]] = self.calculateClassProbs(summaryPerClass, priors, _training_features[i-1])

        _predictions = {}
        for feature_number,log_entries in _training_predictions.iteritems():
            bestLabel, bestProb = None, -1
            for class_num, prob in log_entries.iteritems():
                if bestLabel is None or prob > bestProb:
                    bestProb = prob
                    bestLabel = int(class_num)
            _predictions[int(feature_number)] = bestLabel
        return _predictions

    def calculateProbability(self,x,mean,stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculateClassProbs(self,summary,priors,vector):

        '''
        :param summary: mean and standard deviation for each attribute in each class
        :param priors: Prior probabilities for all our classes
        :param vector: Vector for which likelihood has to be calculated
        :return: returns the loglikelihood calculation of a vector for each class.
        '''

        vector = vector[1:]
        # Calculate class probabilities
        logpriors = {}
        for classval, entry in priors.iteritems():
            logpriors[int(classval)] = math.log(float(entry[0]), math.e)
        #calculate likelihood for each attribute
        loglikelihood = {}
        for classval,entry in logpriors.iteritems():
            loglikelihood[int(classval)] = entry
            for i in range(len(vector)):
                ith_vector = vector[i]
                mean,stdev = summary[int(classval)][i]
                prob = self.calculateProbability(ith_vector,mean,stdev)
                if(prob != 0):
                    loglikelihood[classval] += math.log(prob,math.e)
        return loglikelihood

    def calculateAccuracy(self,predictions,answers):
        _total = len(predictions)
        sum = 0
        for key,value in predictions.iteritems():
            sum += 1 if answers[key] == value else 0
        return (sum/float(_total)) * 100

    def crossvalidation(self,data,numberOfParts):
        #split the data
        partSize = len(data)/numberOfParts
        splittedData = {}
        start = 0
        _global_predictions = {}
        end = partSize
        for i in range(numberOfParts):
            splittedData[i] = data[start:end]
            start = end
            end = end + partSize

        for testdata, training in splittedData.iteritems():
            testpart = splittedData[testdata]
            trainingpart = [splittedData[i] for i in range(len(splittedData)) if i is not testdata]

            #flatten training part, just asthetics
            trainingpart = [item for sublist in trainingpart for item in sublist]
            priors = self.calculatePriors(trainingpart)
            summaryPerClass = self.summarizeByClass(trainingpart)
            _training_predictions = self.predict(testpart, summaryPerClass, priors)
            _global_predictions.update(_training_predictions)
            #print _training_predictions
            answers = {int(sublist[0]): int(sublist[-1]) for sublist in testpart}
            #print answers
            print self.calculateAccuracy(_training_predictions, answers)
            #_training_predictions = self.zeror(testpart,priors)
            #print _training_predictions

            #print self.calculateAccuracy(_training_predictions,answers)
        print _global_predictions
        with open('predictionshw1.csv', 'wb') as f:
            w = csv.writer(f)
            w.writerows(_global_predictions.items())

    def zeror(self,data,priors):
        maxclasslabel =  max(priors,key=priors.get)
        _training_features = [sublist[0:-1] for sublist in data]
        _training_predictions = {}
        for i in range(1,len(_training_features) + 1):
            _training_predictions[_training_features[i-1][0]] = maxclasslabel
        return _training_predictions




def main():
        nb = GaussianNaiveBayes()
        data = '/Users/rakshitsareen/Documents/ML/glasshw1.csv'
        data = nb.loadcsv(data)
        priors = nb.calculatePriors(data)
        summaryPerClass = nb.summarizeByClass(data)
        _training_predictions = nb.predict(data,summaryPerClass,priors)
        answers = {int(sublist[0]) :int(sublist[-1]) for sublist in data}
        nb.calculateAccuracy(_training_predictions,answers)
        nb.crossvalidation(data,5)
        """
        [1.51831, 14.39, 0.0, 1.82, 72.86, 1.41, 6.47, 2.88, 0.0] 0.0125833333333 0.0707640870951 7
        """
        #print nb.calculateProbability(2.88,0.0125833333333,0.0707640870951)

if __name__ == '__main__':
        main()