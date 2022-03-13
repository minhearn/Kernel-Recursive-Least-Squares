import numpy as np
"""
def kernel(v1,v2,sigma=1):
    assert len(v1)==len(v2), "size of v1 and v2 must be equal"
    diff = np.subtract(v1,v2)
    l = len(diff)
    under = 2*(sigma)**2
    over = 0
    for i in range(l):
        over -=diff[i]**2
    return np.exp(over/under)
"""

class KRLS:
    """
    Kernel recursive least-squares
    multi-input single-output
    multi-input multi-output ------------TBD------------
    """
    def __init__(self,x_dim,criterion):
        self.x_dim=x_dim #size of input signal
        #size of output signal------------------TBD-----------------
        self.criterion=criterion #ε0 ALD criterion

        self.dictionary = np.zeros((0,x_dim))    #dictionary
    
    
    def feature(self,x,sigma=10):
        #φ(x) = k(x,.) = exp(-|x-.|^2/2*sigma**2)
        under = -1/(2*sigma**2)
        bandwidth=under
        return np.exp(bandwidth*np.sum((self.dictionary-x)**2,axis=1))

    def dictionary_manage(self,x,y):
        if len(self.dictionary)==0:
            #if there is no element in dictionary, add x into it
            self.dictionary = np.append(self.dictionary,x.reshape(1,-1),axis=0)
            self.K_inv = np.ones((1,1))
            self.P_inv = np.ones((1,1))
            self.params= np.array([y])
        else:
            beta = self.feature(x)  #β=(κ(u1,xn),...κ(uM,xn))
            yhat = np.dot(self.params,beta)
            e = y-yhat
            #Approximate linear dependency
            a = np.dot(self.K_inv,beta) 
            delta = 1 - np.dot(beta,a)
            if delta>self.criterion:
                ####add x into dictionary
                self.dictionary = np.append(self.dictionary,x.reshape(1,-1),axis=0)

                #update K_inv
                self.K_inv = (delta*self.K_inv)+np.dot(a.reshape(-1,1),a.reshape(1,-1))
                self.K_inv = np.append(self.K_inv,-1*a.reshape(1,-1),axis=0)
                self.K_inv = np.append(self.K_inv,np.append(-1*a,1).reshape(-1,1),axis=1)
                self.K_inv = self.K_inv/delta

                #update P_inv
                self.P_inv = np.append(self.P_inv,np.zeros(len(self.P_inv)).reshape(-1,1),axis=1)
                self.P_inv = np.append(self.P_inv,np.append(np.zeros(len(self.P_inv)),1).reshape(1,-1),axis=0)
                
                #update params
                self.params = self.params - a/delta*e
                self.params = np.append(self.params,e/delta)
            else:
                ####not add x into dictionary
                Pa = np.dot(self.P_inv,a)
                q = Pa/(1+np.dot(a,Pa))

                #update P_inv
                self.P_inv = self.P_inv - np.dot(q.reshape(-1,1),Pa.reshape(1,-1))

                #update params
                self.params = self.params + e*np.dot(self.K_inv,q)
        
    def predict(self,x):
        if len(self.dictionary)==0:
            return 0
        else:
            beta = self.feature(x)
            return np.dot(self.params,beta)



