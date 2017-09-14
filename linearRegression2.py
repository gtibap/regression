import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import matplotlib.pyplot as plt

order=2

x = np.array([
0.86,
0.09,
-0.85,
0.87,
-0.44,
-0.43,
-1.10,
0.40,
-0.96,
0.17
])

y = np.array([
2.49,
0.83,
-0.25,
3.10,
0.87,
0.02,
-0.12,
1.81,
-0.83,
0.43
])

x2 = x**2
x3 = x**3

print "x: ",  x
print "x^2:", x2
print "x^3:", x3
print "y: ", y
#print "x.shape[0]: ", x.shape

#############################
######## order ##############
if order==1:
	X = np.stack(( x, np.ones(x.shape[0])))
elif order==2:
	X = np.stack(( x2, x, np.ones(x.shape[0])))
elif order==3:	
	X = np.stack(( x3, x2, x, np.ones(x.shape[0])))
else:
	X = np.stack(( x, np.ones(x.shape[0])))
	
## column shape
X = X.T
print "X:\n ", X
##print "x.T*x: ", np.dot(x, x)

xdx = np.dot(X.T, X)
#print "xdx: ", xdx
ixdx = inv(xdx)
#print "ixdx: ", ixdx

xdy = np.dot(X.T, y)
#print "xdy: ", xdy

w = np.dot(ixdx,xdy)
print "w: ", w

#############################
######## order ##############
if order==1:
	yp = w[0]*x + w[1]
	print "error x0: ", (2.49-(w[0]*(0.86) + w[1]))*(2.49-(w[0]*(0.86) + w[1]))
	
elif order==2:
	yp = w[0]*x2 + w[1]*x + w[2]
elif order==3:
	yp = w[0]*x3 + w[1]*x2 + w[2]*x + w[3]
else:
	yp = w[0]*x + w[1]
	
error = np.sum( (yp-y)**2 )
print "error: ", error

##############
print "######################"
print "##############"

print "second part:"

# initial values for w
#############################
######## order ##############
if order==1:
	w_0 = np.array([1000.0, -1000.0])
elif order==2:
	w_0 = np.array([1000.0, 1000.0, -1000.0])
elif order==3:
	w_0 = np.array([1000.0, 1000.0, 1000.0, -1000.0])
else:
	w_0 = np.array([1000.0, -1000.0])

alpha=0.01
# initial error
w_e = 1.0
#print "w initial: ", w_0
cont=0
while w_e > 0.0001:
	#error gradient
	dE = 2*(np.dot(xdx,w_0) - xdy)
	w_1 = w_0 - alpha*dE	
	w_e = LA.norm(w_1-w_0)
	w_0 = w_1
	cont=cont+1
	print "cont, w_1, w_e: ", cont, ", ", w_1, ", ", w_e	
	
print "result w: ", w_1


## in order to plot we sort x, y, and yp
idx = np.argsort(x)
x = x[idx]
y = y[idx]
yp = yp[idx]
	
plt.plot(x,y, 'ro', x, yp)
plt.show()



