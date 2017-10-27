from __future__ import division
import numpy as np
from numpy import linalg as LA

X = np.array([[1,1,0,1,1],[0,0,1,1,1],[0,1,0,1,0],[1,1,1,1,1]])
Y = np.array([0,0,0,1,1])

test = np.array([0,0,0,1])

X=X.T
print "X:\n", X 
print "X[0]: ", X[0]

order=3 # number of features
alpha=0.01
error=10
acc=np.zeros(order+1)
w_i = np.array([100.0, 110.0, 230.0, -450.0])

cont=0
while error > 0.0001:
	# label: 0 or 1
	for (sample, label) in zip(X,Y):
		#print "sample: ", sample
		wTx=np.dot(sample,w_i)
		#print "wTx:",wTx
		sigma = 1/(1+np.exp(-wTx))
		#print "sigma: ", sigma
		acc+=np.dot(sample, (label-sigma))
		#print "acc:\n",acc
	w_out = w_i + np.dot(alpha,acc)
	error = LA.norm(w_out - w_i)
	#print "error: ",error
	w_i = w_out
	cont+=1
	if cont > 10:
		alpha=alpha*0.9
		#print "alpha, error: ", alpha,", ", error
		cont=0

print "w:\n",w_out	
wTx=np.dot(test,w_out)
sigma = 1/(1+np.exp(-wTx))
print "p(y=1): ", sigma
