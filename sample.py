# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:45:07 2020

@author: narul
"""

from environment import MountainCar
import sys
import numpy as np

def sparse_dot(x,s):
    product=0
    #a=[]
    for i in range(len(x)):
        if (i not in s) :
            continue
        
         #   continue
        if s[i]!=0:
            product+=s[i]*x[i]
    return product       

def cons(s,c):
    for i in s:
        s[i]=c*s[i]
    return s

def diff(a,s):
    for i in range(len(a)):
        if i not in s:
            continue
        a[i]=a[i]-s[i]
    return a    




alpha=0.01
gamma=0.99
epsilon=0
episodes=4
max_iter=2



new=MountainCar(mode='raw')
    #initialization of parameters
theta=np.zeros((new.action_space,new.state_space))  #3 actions and 2 states
bias=0


new.state=new.reset()




for i in range(10):
    if np.random.random()<1-epsilon:
        action=0
        x=sparse_dot(theta[action], new.state)+bias
        #print(x)
        for j in range(new.action_space):
                   if   (sparse_dot(theta[j], new.state)+bias)>x:
                      x=sparse_dot(theta[j], new.state)+bias
                      opt_action=j
        action=opt_action
    else:
        action = np.random.randint(0, 3)
    
    print(action)
    f=cons(new.state, alpha)
    
    m=sparse_dot(theta[action], new.state)+bias
    new.state, reward, done=new.step(action)        
    
    sprime=new.state
#if done==True:
 #   break
#print(state2)
#reached sprime, now take the best action find y
    x=sparse_dot(theta[action], sprime)+bias
    for j in range(new.action_space):
        if   (sparse_dot(theta[j], sprime)+bias)>x:
            x=sparse_dot(theta[j], sprime)+bias
            opt_action=j
    y=reward+gamma*x
    TD=-(y-(m))

    theta[action]=(diff(theta[action],cons(f,TD)))
    bias=bias-TD*alpha

    
print(theta)    


