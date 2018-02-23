import numpy as np
import matplotlib.pyplot as plt
from data import *
class ANN():
    def __init__(self, X, y, parameters, learning_rate, n_iter, lamda):
        #data set
        self.X=X
        #laebels
        self.y=y
        #learning rate
        self.learning_rate=learning_rate
        #number of iterations
        self.epoch = n_iter
	self.lamda=lamda
        #number of neurons in the input layers
        self.n_layers = len(parameters)
        #Counts number of nodes in each layer
        self.sizes = [layer[0] for layer in parameters]
        #Activation functions for each layer.
        self.fs =[layer[1] for layer in parameters]
        #Derivatives of activation functions
        self.fprimes = [layer[2] for layer in parameters]
        #builds the network
        self.setup_network()
        #contains train, validation and test errors, respectively
	self.train_error = []
	self.valid_error = []
	self.test_error = []
       
 
    def setup_network(self):
        #weghit of the layers
        self.weights=[]
        #gradient of weight
        self.w_grad = []
        #gradient of bias
        self.b_grad = []
        # bias, input, outputs vectors of layers
        self.biases=[]
        self.inputs=[]
        self.outputs=[]
        #error at each layer
        self.delta=[]
        ##weights are randomly initialized, biased are initialized by 1
        for layer in range(self.n_layers-1):
            n = self.sizes[layer]
            m = self.sizes[layer+1]
            np.random.seed(0)
            self.weights.append(np.random.normal(0,0.01, (m,n)))
            self.biases.append(np.random.normal(0,0.01,(m,1)))
            #partial derivatives for weights and biases whic are initialized to zero
            self.w_grad.append(np.zeros((m,n)))
            self.b_grad.append(np.zeros((m,1)))
            self.inputs.append(np.zeros((n,1)))
            self.outputs.append(np.zeros((n,1)))
            #delta is the error term 
            self.delta.append(np.zeros((n,1)))
            
        #The last layer are initialized to zero separately
        n = self.sizes[-1]
        self.inputs.append(np.zeros((n,1)))
        self.outputs.append(np.zeros((n,1)))
        self.delta.append(np.zeros((n,1)))
	
	
    #gets the data and propagates the data from input to the output layer
    def feedforward(self, x):
        #chacks the dimension of the data 
        if x.ndim == 1:
            x.resize(len(x),1)
        self.inputs[0]=x  #Just for  clarification
        self.outputs[0]=x
        #multipling the input data by the weithts and biased
        for i in range(1,self.n_layers):
            self.inputs[i]=np.dot(self.weights[i-1],self.outputs[i-1])+self.biases[i-1]
            self.outputs[i]=self.fs[i](self.inputs[i])
	#return the output which is s single number		
        return self.outputs[-1]
   #updates the gradients for weights and biased
    def update_gradient(self,x,y):
        output = self.feedforward(x)
        self.delta[-1] =self.fprimes[-1](self.outputs[-1])*(output-y)
 				
        n=self.n_layers-2
        for i in xrange(n,0,-1):
            #computes the error term
            self.delta[i] = self.fprimes[i](self.inputs[i])*np.dot(self.weights[i].T, self.delta[i+1])
            #Compute the desired partial derivatives for weights and biased
            self.w_grad[i]+=np.outer(self.delta[i+1],self.outputs[i])
            self.b_grad[i]+= self.delta[i+1]
        #Compute the desired partial derivatives for weights and biased in the first layer
        self.w_grad[0]+= np.outer(self.delta[1],self.outputs[0])
        self.b_grad[0]+=self.delta[1]
    #since we are using batch processing at each
    #iteration set the gradient and biased to zero     
    def set_w_to_zero(self):
        for i in xrange(self.n_layers-1):
           self.w_grad[i]=0
           self.b_grad[i]=0 
     
   #updating weights and biased
    def update_weights(self):
        n=self.n_layers-2
        k=(1.0/len(self.y))
        lr = self.learning_rate
        for i in xrange(n,0,-1):
           #updates the weights and the biased
            self.weights[i] = self.weights[i] - k*lr*self.w_grad[i] -lr*self.lamda*self.weights[i]
            self.biases[i] = self.biases[i] - k*lr*self.b_grad[i]
        #updating the first layer
        self.weights[0] = self.weights[0] - k*lr*self.w_grad[0]-lr*self.lamda*self.weights[0]
        self.biases[0] = self.biases[0] - k*lr*self.b_grad[0]	
	
    #training the model
    def train(self, data):
        for repeat in range(self.epoch):
            for row in range(len(self.X)):
                x=self.X[row]
                y=self.y[row]
                self.update_gradient(x,y)
            #updates the weights
            self.update_weights()
            #set gradients to zero
            self.set_w_to_zero()

            #calculates the error for traing, validation and testing set, respectively
            self.train_error.append(cost(self.predict(data.tr_set), data.tr_labels))
            self.valid_error.append( cost(self.predict(data.val_set), data.val_labels))
            self.test_error.append( cost(self.predict(data.ts_set), data.ts_labels))
            print repeat, np.column_stack((self.train_error[-1], self.valid_error[-1], self.test_error[-1]))
	return self.train_error, self.valid_error, self.test_error, self.predict(data.tr_set), self.predict(data.val_set), self.predict(data.ts_set)
   
    #predicts the targets of testing set
    def predict(self, X):
        n = len(X)
        m = self.sizes[-1]
        prediction = np.empty((n,m))
        for i in range(len(X)):
            prediction[i] = self.feedforward(X[i])
			
        return prediction
#sigmoid function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
#gradient of sigmoid function 
def sigmoid_grad(x):
    return sigmoid(x)*(1-sigmoid(x))
#identity function
def identity(x):
    return x
#gradient of identity functio which equals to 1 
def identity_grad(x):
    return 1
#calculating sum of square errors
#takes two inputs, actual and the predicted labels
def cost(predicted, labels):
    return np.sum((labels - predicted)**2)/len(labels) 
	





