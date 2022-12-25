
import numpy as np
from numpybd.loss import * 

def train_NMF(V, max_iter, epsilon, num_basis, beta):
    
    # Initilaze the basis and weight vectors
    B = np.random.rand(V.shape[0], num_basis) 
    W = np.random.rand(num_basis, V.shape[1]) 
       
    # Multiplicative rules for NMF   
    for ii in range(max_iter):
        B_t = np.transpose(B)
        W_t = np.transpose(W)
        
        B = B * np.matmul(V*np.power(np.matmul(B, W), (beta - 2)), W_t)/np.matmul(np.power(np.matmul(B, W), (beta - 1)),  W_t)
        W = W * np.matmul(B_t, (V*np.power(np.matmul(B, W), (beta - 2))))/np.matmul(B_t, np.power(np.matmul(B,W), (beta - 1))) 
        V_tilde = np.matmul(B,W)
        
        cost = get_cost(V, V_tilde, beta) # get the cost for each different cost functions
        print("Training: Iteration: %03d, cost: %s" %(ii+1, cost))

        if cost < epsilon:
            break
    return B, W

    
def test_NMF(V, B_train, max_iter, epsilon, num_basis, beta):       
    
    # Initilaze the basis and weight vectors, but for the test stage, basis matrix is fixed
    W = np.random.rand(num_basis, V.shape[1])  #size: (basis,time)
    B = B_train
    
    B_t = np.transpose(B)
    
    # Multiplicative rules for NMF   
    for i in range(max_iter):
        
        W = W * np.matmul(B_t, (V*np.power(np.matmul(B, W), (beta - 2))))/np.matmul(B_t, np.power(np.matmul(B,W), (beta - 1))) 
        V_tilde = np.matmul(B, W)
        
        cost = get_cost(V, V_tilde, beta) # KL divergence
        print("Test: Iteration: %03d, cost: %s" %(i+1, cost))
        if cost < epsilon:
            break
    return W, cost 

        
        
def get_cost(V, V_tilde, beta):
    if beta == 0:# Euclidean Distance
        cost = beta_div(V, V_tilde, beta, reduction='mean') # beta_divergence cost function is implemented from numpybd library
    elif beta == 1: # KL Divergence
        cost = np.linalg.norm(V - V_tilde, 'fro') 
    elif beta == 2: # IS Divergence
        cost = np.linalg.norm(V - V_tilde, 'fro') 
    else:
        cost = 0
        
    return cost