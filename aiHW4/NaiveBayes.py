import numpy as np
from Methods import Predictor


class NaiveBayes(Predictor):

    def __init__(self):
        self.likelihoods = {}
        self.class_priors = {}

    def train(self, instances):
        classes = {}
        # Data pre-processing
        for instance in instances:
            label = str(instance.label)
            if label not in self.class_priors:
                self.class_priors[label] = 0
                self.likelihoods[label] = {}
                classes[label] = []
            classes[label].append(instance)
            self.class_priors[label] += 1
        # print('likelihoods', self.likelihoods)
        # print('classes', classes)
        # print('class_priors', self.class_priors)
        features = instances[0].feature_vector.feature_vec.keys()
        # Training model
        for label in self.class_priors:
            self.class_priors[label] = self.class_priors[label] / float(len(instances))
            for feature in features:
                mean = self._mean(classes[label], feature)
                print('Label: ' + str(label) + ' mean: ' + str(mean))
                variance = self._variance(classes[label], feature, mean)
                print('Label: ' + str(label) + ' variance: ' + str(variance))
                self.likelihoods[label][feature] = mean, variance

    def _mean(self, instances, feature):  # Can be improved with np
        s = 0
        for instance in instances:
            s += instance.feature_vector.feature_vec[feature]
        return (1/float(len(instances))) * s

    def _variance(self, instances, feature, mean):  # Can be improved with np
        sl = 0
        for instance in instances:
            sl += (instance.feature_vector.feature_vec[feature] - mean) ** 2
        return (1/float(len(instances) - 1)) * sl

    def _gaussian(self, value, mv):
        mean, variance = mv
        # mean = 0.8551401869158878
        # variance = 0.9507481023210923
        # value = 2.0
        # print('value', value)
        # print('m, v', mv)
        # gaussian = (1/np.sqrt(2 * np.pi * (variance))) * np.exp((-1/(2*(variance)))*(value - mean))
        gaussian = (1/np.sqrt(2*np.pi*variance)) * np.exp((-1/(2*variance))*(value - mean)**2)
        # print('gaussian', gaussian)
        return gaussian

    def predict(self, instance):
        posteriors = {}
        features = instance.feature_vector.feature_vec.keys()
        for label in self.class_priors:
            posteriors[label] = self.class_priors[label]
            for feature in features:
                posteriors[label] *= self._gaussian(instance.feature_vector.feature_vec[feature], self.likelihoods[label][feature])
        return max(posteriors, key=posteriors.get)
