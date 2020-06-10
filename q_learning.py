# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:18:41 2020

@author: narul
"""
import numpy as np
from environment import MountainCar
import sys
mod=sys.argv[1]
weight_out=sys.argv[2]
reward=sys.argv[3]
episodes=sys.argv[4]
max_iter=sys.argv[5]
epsilon=sys.argv[6]
gamma=sys.argv[7]
alpha=sys.argv[8]

def dictv(s, a):
    c=[]
    for i in range(a):
        if i not in s:
           c.append(0)
        else:
            c.append(s[i])
    npa=np.array(c)
    return npa

weight_out=open(weight_out, 'w')
reward_out=open(reward, 'w')
    


car=MountainCar(mode=mod)

theta=np.zeros((car.action_space, car.state_space))
bias=0
q=np.zeros((car.state_space,car.action_space))

rewar=[]
for e in range(int(episodes)):
    state1=car.reset()
    rew=0
    for i in range(int(max_iter)):
        
        sv=dictv(state1, car.state_space)
        for a in range(car.action_space):
            q[0,a]=np.matmul(sv, theta[a])+bias
        
        if np.random.random()<1-float(epsilon):
            action=np.argmax(q[0])
        else:
            action=np.random.randint(0,3)
            
            
        y_predict=np.matmul(sv, theta[action])+bias    
        real_state=float(alpha)*sv    
        
        
        
        state2, reward, done=car.step(action)  
        state1=state2
        sv2=dictv(state2, car.state_space)
        for a in range(car.action_space):
            q[1,a]=np.matmul(sv2, theta[a])+bias 
        aprime=np.argmax(q[1])
        y=reward+float(gamma)*(np.matmul(sv2, theta[aprime])+bias)
        
        TD=-(y-y_predict)
        
        theta[action]=theta[action]-TD*real_state
        bias=bias-TD*float(alpha)
        rew+=reward
        
        if done==True:
            break
    reward_out.write(str(rew))
    reward_out.write("\n")
        

weight_out.write(str(bias))
weight_out.write("\n")
for i in range(theta.shape[1]):
    for j in range(theta.shape[0]):
        
        weight_out.write(str(theta[j,i]))
        weight_out.write("\n")
        
        
