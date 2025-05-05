import numpy as np

x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
X=x/np.amax(x,axis=0)
y=y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return x*(1-x)
epoch=1000
learning_rate=0.6
inputlayer_neuron=2
hiddenlayer_neuron=3
output_neuron=1

wh=np.random.uniform(size=(inputlayer_neuron,hiddenlayer_neuron))
bh=np.random.uniform(size=(1,hiddenlayer_neuron))
wo=np.random.uniform(size=(hiddenlayer_neuron,output_neuron))
bo=np.random.uniform(size=(1,output_neuron))

for i in range(epoch):
    net_h=np.dot(x,wh)+bh
    sigma_h=sigmoid(net_h)
    net_o=np.dot(sigma_h,wo)+bo
    output=sigmoid(net_o)
    deltak=(y-output)*derivatives_sigmoid(output)
    deltah=deltak.dot(wo.T)*derivatives_sigmoid(sigma_h)
    wo=wo+sigma_h.T.dot(deltak)*learning_rate
    wh=wh+x.T.dot(deltah)*learning_rate
    
print("Input \n"+str(x))
print("Actual Output \n"+str(y))
print("Predicited Output: \n",output)

Input 
[[0.66666667 1.        ]
 [0.33333333 0.55555556]
 [1.         0.66666667]]
Actual Output 
[[92.]
 [86.]
 [89.]]
Predicited Output: 
 [[1.]
 [1.]
 [1.]]
