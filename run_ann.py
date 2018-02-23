from batch import *
#from minibatch import *
from data import data_preparation
import numpy as np
import matplotlib.pyplot as plt

#prepares the data, splits the data into training, validation and testing set and 
#their corresponding labels
dataset = data_preparation()
print 'Data created.'
train_set = dataset.tr_set
train_labels = dataset.tr_labels 

#validation set and labels
valid_set = dataset.val_set
valid_labels = dataset.val_labels

#test set and labels
test_set = dataset.ts_set
test_labels = dataset.ts_labels

#parameters of the neural network

#each prantheses contains number of neurons, activation function and the derivative
#of the activation function of the corresponding layer
#for exmaple the hidden layer contains 100 neurons and sigmoid activation function
param=((len(train_set[0]),0,0),(100, sigmoid, sigmoid_grad),(1,identity, identity_grad))

#build the model
N=ANN(train_set, train_labels, param, learning_rate=.001, n_iter=1200, lamda=.9)
#training the model and evaluation error on training, validation and testing data
train_error, valid_error, test_error, train_pred, valid_pred, test_pred = N.train(dataset)

#plots error on the testing data
plt.plot(test_error, 'r', label = 'test')
plt.legend('test')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()
#plots the prediction on the testing data
act=np.cumsum(test_labels)
#pred=np.cumsum(test_pred[1:])
pred=test_pred[1:]
#ppred=np.zeros(1)
ppred=[]
ppred.append(act[0])
for i in range(1,len(act)-1):
	ppred.append(pred[i]+act[i-1])
print len(act), len(ppred)

plt.plot(act,'-b', ppred,'r', label=('Actual', 'Predicted'))
plt.legend(('Actual', 'Predicted'))
plt.xlabel('Day')
plt.ylabel('Price')

plt.show()

'''print 'training the model done!'

#predictions on training, validation and testing data
train_prediction = N.predict(train_set)
valid_prediction = N.predict(valid_set)
test_prediction = N.predict(test_set)

idx = np.random.randint(0,700, 20)
print 'train comparison'
print np.column_stack((train_labels[idx], train_prediction[idx]))
print 'validation comparison'
print np.column_stack((valid_labels[idx], valid_prediction[idx]))
print 'test comparison'
print np.column_stack((test_labels[idx], test_prediction[idx]))
#print np.shape(tr_prediction),  np.shape(val_prediction), np.shape(ts_prediction)

train_error = cost(tr_prediction, train_labels)
valid_error = cost(valid_prediction, valid_labels)
test_error = cost(test_prediction, test_labels)

print 'train error is: %f', train_error
print 'validation error is: %f', valid_error
print 'test error is: %f', valid_error'''
	



