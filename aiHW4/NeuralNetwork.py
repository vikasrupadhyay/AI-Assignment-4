from Methods import Predictor, ClassificationLabel
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 - np.exp(-x))

def dsigmoid(x):
    return (x * (1.0 - x)) #TODO TEST

class NeuralNetwork(Predictor):

    def __init__(self):
        self.inputs = 0
        self.hidden = 0
        self.outputs = 0

        self.labels = {}

        self.ai = []
        self.ah = []
        self.ao = []
        self.wi = []
        self.wo = []

    def train(self, instances):
        label_list = []
        for instance in instances:
            if str(instance.label) not in label_list:
                label_list.append(str(instance.label))

        tmp = 0
        for label in label_list:
            self.labels[label] = []
            for i in range(len(label_list)):
                if tmp == i:  # TODO TEST
                    self.labels[label].append(1)
                else:
                    self.labels[label].append(0)
            tmp += 1

        self.inputs = len(instances[0].feature_vector.feature_vec) + 1
        self.hidden = self.inputs - 1
        self.outputs = len(label_list)

        self.ai = [1.0] * self.inputs
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.outputs

        n = 1.0 / np.sqrt(self.inputs)
        self.wi = np.random.uniform(-n, n, (self.inputs, self.hidden))
        self.wo = np.random.uniform(-n, n, (self.hidden, self.outputs))

        learning_rate = 0.15
        epochs = 100

        for epoch in range(epochs):
            np.random.shuffle(instances)
            for instance in instances:
                self._feed_forward(instance)
                self._backwards_propagate(self.labels[str(instance.label)], learning_rate)

    def _feed_forward(self, instance):

        for i in range(self.inputs - 1):
            self.ai[i] = instance.feature_vector.feature_vec[i]

        for i in range(self.hidden):
            total = 0.0
            for j in range(self.inputs):
                total += self.ai[j] * self.wi[j][i]
            # TODO TEST

        for i in range(self.outputs):
            total = 0.0
            for j in range(self.hidden):
                total += self.ah[j] * self.wo[j][i]
            self.ao[i] = sigmoid(total)

    def _backwards_propagate(self, target, rate):

        do = [0.0] * self.outputs
        for i in range(self.outputs):
            error = target[i] - self.ao[i]
            do[i] = dsigmoid(self.ao[i]) * error

        dh = [0.0] * self.inputs
        for i in range(self.hidden):
            error = 0.0
            for j in range(self.outputs):
                error += do[j] * self.wo[i][j]
            dh[i] = dsigmoid(self.ah[i]) * error

        for i in range(self.hidden):
            for j in range(self.outputs):
                self.wo[i][j] += rate * do[j] * self.ah[i]

        for i in range(self.inputs):
            for j in range(self.hidden):
                self.wi[i][j] += rate * dh[j] * self.ai[i]

    def predict(self, instance):
        self._feed_forward(instance)

        index = 0
        for i in range(len(self.ao)):
            if self.ao[index] < self.ao[i]:
                index = 1

        result = []
        for i in range(len(self.ao)):
            if i == index:
                result.append(1)
            else:
                result.append(0)

        for label in self.labels:
            if result == self.labels[label]:
                return ClassificationLabel(label)
