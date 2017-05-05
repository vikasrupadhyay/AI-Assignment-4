from Methods import *
import numpy as np


class NeuralNetwork(Predictor):
    input_batch = 0
    input_labels = []
    num_examples = 0  # training set size
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality

    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength
    class_labels = []
    class_label_map = {}
    model = 0

    def __init__(self):
        self.num_examples = 0  # training set size
        self.nn_input_dim = 2  # input layer dimensionality
        self.nn_output_dim = 2  # output layer dimensionality
        self.epsilon = 0.0001  # learning rate for gradient descent
        self.reg_lambda = 0.01  # regularization strength
        self.class_labels = []
        self.class_label_map = {}
        self.input_batch = 0
        self.input_labels = []
        self.model = 0
        self.input_label_categories = []

    def calculate_loss(self, model):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation to calculate our predictions
        z1 = self.input_batch.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(self.num_examples), self.input_label_categories])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / self.num_examples * data_loss

    def build_model(self, nn_hdim, num_passes=10000):

        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(1234)
        W1 = np.random.randn(self.nn_input_dim, nn_hdim) / np.sqrt(self.nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, self.nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, self.nn_output_dim))

        # This is what we return at the end
        model = {}

        # Gradient descent. For each batch...
        for i in xrange(0, num_passes):

            # Forward propagation
            z1 = self.input_batch.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(self.num_examples), self.input_label_categories] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(self.input_batch.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * W2
            dW1 += self.reg_lambda * W1

            # Gradient descent parameter update
            W1 += -self.epsilon * dW1
            b1 += -self.epsilon * db1
            W2 += -self.epsilon * dW2
            b2 += -self.epsilon * db2

            # Assign new parameters to the model
            model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        print "Training Successful"
        return model

    def train(self, instances):
        temp_toto = []
        for instance in instances:
            if instance.label.label_str not in self.class_labels:
                self.class_labels.append(instance.label.label_str)
                self.class_label_map[len(self.class_labels) - 1] = instance.label.label_str
            self.input_labels.append(instance.label.label_str)
            temp = []
            for i in xrange(0, len(instance.feature_vector.feature_vec)):
                temp.append(instance.feature_vector.feature_vec[i])
            temp_toto.append(temp)
            self.nn_input_dim = len(temp)
            self.nn_output_dim = len(self.class_labels)
        self.input_batch = np.array(temp_toto)
        self.num_examples = len(self.input_batch)
        for label in self.input_labels:
            for i in xrange(0, len(self.class_label_map)):
                if label == self.class_label_map[i]:
                    self.input_label_categories.append(i)
                    break
        self.model = self.build_model(20)

    def make_prediction(self, x):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # print probs
        return np.argmax(probs, axis=1)

    def predict(self, instance):
        temp = []
        for i in xrange(0, len(instance.feature_vector.feature_vec)):
            temp.append(instance.feature_vector.feature_vec[i])
        return self.class_label_map[self.make_prediction(np.array(temp))[0]]