from mnist import MNIST
import numpy as np
np.random.seed(42)

class Layer:
    def __init__(self, layer_index, number_of_neurons, number_of_neurons_in_previous_layer, is_output_layer):
        self.layer = layer_index
        self.is_output_layer = is_output_layer
        self.size = number_of_neurons

        self.weights = np.random.rand(number_of_neurons, number_of_neurons_in_previous_layer) * 0.01 # generates a matrix of random floats [0, 1)
        self.activation = None
        self.bias = np.zeros(number_of_neurons)

    def activation_function(self, z): 
        return 1/(1 + np.exp(-z)) # sigmoid
    def activation_derivative(self, a):
        return a * (1 - a) # sigmoid derivative
    
    def gradient(self, prev_activation: np.ndarray, dEda1): 
        ''' 
        Returns: gradients (dEdw, dEdb, dEda0) which is on the form (matrix, vector, matrix)
            a0 means activation for layer to the left
            a1 means activation for current layer

        dEdw = derivative of error with respect to weight
        E=Total Error; w=weight; a=activation; z=linear combination, L=layer;
    
        dEda1 = gradient to the right of me
            for output layer: 2(a-y)
            for hidden layer: dE

        For output neuron: wL --> zL --> aL --> Etotal
        For rightmost hidden neuron: w(L-1) --> z(L-1) --> a(L-1) --> zL...L --> aL --> Etotal
            dEdwL = dEdaL * (daLdzL) * dzLdwL
            dEdwL-1 = dEdaL * daLdzL * dzLdaL-1 * (daL-1dzL-1) * dzL-1dwL-1 = dEdwL / dzLdwL
        '''

        # z0 --> z1=current --> z2
        # gradient for weights
        dadz = self.activation_derivative(self.activation)
        dzdw = prev_activation
        dEdz = dEda1 * dadz
        dEdw = np.outer(dEdz, dzdw) # matrix

        # gradient for bias
        dEdb = dEdz # vector

        # gradient to calculate further backprop
        dzda0 = self.weights.T #?
        dEda0 = np.dot(dzda0, dEdz) # matrix

        return (dEdw, dEdb, dEda0)

# LOAD DATA
mndata = MNIST('./data')
images_train, labels_train = mndata.load_training() 
images_test, labels_test = mndata.load_testing() 
images_train = (np.array(images_train, dtype=np.float32)) / 255.0 # convert to numpy arrays and normalize pixel values to [0, 1]
labels_train = np.array(labels_train, dtype=np.int64)
images_test = (np.array(images_test, dtype=np.float32)) / 255.0 
labels_test = np.array(labels_test, dtype=np.int64)

# 6000 training examples where each image is 28x28=784 images
print(f"We have loaded {len(images_train)} images to train. There are {len(images_train[0])} input neurons")

def cost_function(arA, arB): # using MSE (mean squared error as cost function)
    MSE = np.square(arA-arB)
    return MSE

def forward_pass(input_layer): 
    # returns (output_layer, prediction)
    previous_activation = input_layer
    for layer in nn[1:]: 
        z = np.matmul(layer.weights, previous_activation) + layer.bias # calculates z1 = W*(a0) + b1
        layer.activation = layer.activation_function(z) 
        previous_activation = layer.activation
    
    pred = np.argmax(previous_activation) # finds index of max element
    return (previous_activation, pred)

# INITIALIZATION - small weights and biases
lr = 0.1 # learning rate
neurons_in_layer = [784, 200, 10] 
nn  = [images_train[0]]
for i in range(1, len(neurons_in_layer)):
    nn.append(Layer(i, neurons_in_layer[i], neurons_in_layer[i-1], i==len(neurons_in_layer)-1))

# TRAINING
epochs = 10
for epoch in range(epochs):
    total_cost = 0
    total_examples = 6000
    correct = 0
    for example in range(total_examples): 
        # FORWARD PASS 
        (output_layer, pred) = forward_pass(images_train[example])
        
        # CALCULATE ERROR
        target = np.zeros(10)
        target[labels_train[example]] = 1
        cost = cost_function(output_layer, target)
        total_cost += np.sum(cost)
        if pred == labels_train[example]:
            correct += 1
        
        # BACK PROPAGATION - with gradient descent
        dEda1 = []
        for i in range(len(nn)-1, 0, -1):
            layer = nn[i]
            previous_activation = images_train[example] if i==1 else nn[i-1].activation
            if layer.is_output_layer:
                dEda1 = 2 * (layer.activation - target) 
                (dEdw, dEdb, dEda0) = layer.gradient(previous_activation, dEda1)
            else:
                (dEdw, dEdb, dEda0) = layer.gradient(previous_activation, dEda1)
            dEda1 = dEda0
                
            # update weights and biases, and not use updated values for further backprop
            layer.weights -= lr * dEdw 
            layer.bias -= lr * dEdb
    accuracy = correct / total_examples * 100
    print(f"Epoch: {epoch} | Cost: {total_cost} | Accuracy: {accuracy}%")
    

# EVALUATE MODEL
correct = 0
total_examples = 6000
for example in range(total_examples):
    (output_layer, pred) = forward_pass(images_test[example])
    if labels_test[example] == pred:
        correct += 1
accuracy = correct / total_examples * 100 
print(f"Accuracy: {accuracy}%")


# TODO: decreasing learning rate


            

            




        
        
