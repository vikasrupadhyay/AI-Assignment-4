from abc import ABCMeta, abstractmethod
import numpy as np

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        #self.label_num = int(label)
        self.label_str = str(label)
        pass
        
    def __str__(self):
        print self.label_str
        pass

# the feature vectors will be stored in dictionaries so that they can be sparse structures
class FeatureVector:
    def __init__(self):
        self.feature_vec = {}
        pass
        
    def add(self, index, value):
        self.feature_vec[index] = value
        pass
        
    def get(self, index):
        val = self.feature_vec[index]
        return val
        

class Instance:
    def __init__(self, feature_vector, label):
        self.feature_vector = feature_vector
        self.label = label

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

"""
TODO: you must implement additional data structures for
the three algorithms specified in the hw4 PDF

for example, if you want to define a data structure for the
DecisionTree algorithm, you could write

class DecisionTree(Predictor):
	# class code

Remember that if you subclass the Predictor base class, you must
include methods called train() and predict() in your subclasses
"""



class DecisionTree(Predictor):
    def main():


        return 0


class NeuralNetwork(Predictor):

    def __init__(self):
        l1_w =[]
        l2_w =[]

        self.l1_w =[]
        self.l2_w=[]

        


    def train(self, instances):
        for instance in instances:
            #print instance._feature_vector.feature_vec
            #print  instance._label.label_str

            X=[]

        #creating training arrays

        for instance in instances:
            temp=[]

            for key,value in instance._feature_vector.feature_vec.iteritems():

                temp.append(value)

            X.append(temp)
        #print X[1]

        X = np.array(X)

        Y=[]



        for instance in instances:
            Y.append(instance._label.label_str)



        # Standardizing the labels for prediction ( cant predict on strings for basic reasons, Input is numeric, labels are strings of characters)
        commons=[]
        for i in Y:
            if i not in commons:
                commons.append(i)
        #print" The length of the commons", commons

        for count in range(0, len( Y)):
            #print count
            for c in range(0,len(commons)):
                if Y[count] == commons[c]:
                    Y[count]= c

        print Y

        for i in range(0,len(Y)):
            temp=[]
            temp.append(Y[i])
            Y[i]=temp
        Y =np.array(Y)


        num_of_features = len(X[0])
        input_layer_size = len(X)
        number_of_labels = len(Y)
        hidden_layers = 2
        np.random.seed(1)

        #random Initlializing of weights of the two hidden layers
        theta0 = np.random.random((num_of_features, input_layer_size) ) * 2  - 1
        theta2 = np.random.random((number_of_labels, 1)) * 2  - 1

        #For a dual hidden layer setting

        for iter in range(0,60000):
            # Propogation
            l0 = X
            prod = np.dot(l0,theta0)
            l1 = 1 / (1 + np.exp(-prod)) 
            prod = np.dot(l1,theta2)
            l2 = 1 / (1 + np.exp(-prod)) 

            l2error = l2 - Y

            #calculation of error
        
            if (iter% 10000) == 0:
                print "Error after "+str(iter)+" iterations:" + str(np.mean(np.abs(l2error)))

            l2delt = l2error*(l2 * (1-l2))

            l1error = l2delt.dot(np.transpose(theta2))
            l1delt = l1error * (l1*(1-l1))

            #Update the weights
            
            theta2 = theta2 - 0.001*(l1.T.dot(l2delt))
            theta0 = theta0 - 0.001*(l0.T.dot(l1delt))
        self.l1_w = theta0
        self.l2_w = theta2

        print len(theta0),len(theta2),len(X)

    """      For a Single hidden layer setting  

            for iter in range(0,80000):

            l0 = X
            prod = np.dot(l0,theta0)
            l1 = 1 / (1 + np.exp(-prod)) 
            prod = np.dot(l1,theta2)
            #l2 = 1 / (1 + np.exp(-prod)) 

            l1error = l1 - Y
        
            if (iter% 10000) == 0:
                print "Error after "+str(iter)+" iterations:" + str(np.mean(np.abs(l2error)))

            l1delt = l1error*(l1 * (1-l1))

            #l1error = l2delt.dot(np.transpose(theta2.T))
            #l1delt = l1error * (l1*(1-l1))
            
            #theta2 = theta2 - 0.001*(l1.T.dot(l2delt))
            theta0 = theta0 - 0.001*(l0.T.dot(l1delt))
    
    """
    def predict(self, instance):

        print instance._feature_vector.feature_vec

        X=[]

        for i in instance._feature_vector.feature_vec :
            X.append(i)

        l1_op = 1/(1 +(- np.dot(np.array(X),self.l1_w)))
        l2_op = 1 / ( 1 + (-np.dot(l1_op,self.l2_w)))

        print l2_op



class NaiveBayes():

    def main():

        return 0
