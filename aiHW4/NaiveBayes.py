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
        print('likelihoods', self.likelihoods)
        print('classes', classes)
        print('class_priors', self.class_priors)

        features = instances[0].feature_vector.feature_vec.keys()
        def get_feature_vector(x):
            return x.feature_vector.feature_vec

        # Training model
        for label in self.class_priors:
            self.class_priors[label] = self.class_priors[label] / float(len(instances))
            class_features = list(map(get_feature_vector, classes[label]))
            print('class-features', label, class_features)
            for feature in features:
                mean = self._mean(class_features, feature)
                variance = self._variance(classes[label], feature, mean)
                self.likelihoods[label][feature] = mean, variance

    def _mean(self, instances, feature):
        return np.mean(instances, axis=0)

    def _variance(self, instances, feature, mean):
        return np.var(instances)

    def predict(self, instance):
        posteriors = {}
        for label in self.class_priors:
            pass
