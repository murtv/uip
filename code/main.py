import math
import pickle
import random
import csv
from itertools import islice

# constants

LAYERS = [28, 2, 1] # layer sizes
LEARN_RATE = 0.9
LEARN_ITERS = 1000 # learn iterations

DATASET_FILE = 'dataset.csv'
# map classes to target prediction values
CLASSES = {
    'Kirmizi_Pistachio': 0,
    'Siirt_Pistachio': 1
}

# get table with random values between 0 and 1
def rand_table(rows, cols):
    return [[random.uniform(0, 1) for col in range(cols)] for row in range(rows)]

# a 3d list of weight tables, initialized with 0 - 1 random values
# the weight table at index l stores weights connecting layers l and l + 1
# weight connecting node i in layer l to node j in layer l + 1 is at weights[l][i][j]
weights = [rand_table(LAYERS[layer], LAYERS[layer+1]) for layer in range(len(LAYERS)-1)]
# weights = None

# util functions

# normalize value between 0 and 1
def norm(val, top, bot):
    return (val - bot) / (top - bot)

# normalize values of a table
def norm_tab(tab):
    for col in range(len(tab[0])):
        column = getcol(tab, col)
        top = max(column)
        bot = min(column)
        for row in range(len(tab)):
            tab[row][col] = norm(tab[row][col], top, bot)

# parse dataset file row
def prep_row(row):
    feats = [float(feat) for feat in row[:-1]]
    label = CLASSES[row[-1]]
    return feats, label

# load data from csv file
def load_dataset():
    with open(DATASET_FILE, newline='') as f:
        reader = csv.reader(f)
        next(reader) # skip first row (its the col names)
        return [prep_row(row) for row in reader]
        
# normalize dataset. dataset is list of (feats, targ) tuples
def norm_dataset(dataset):
    all_feats = [feats for feats, _ in dataset]
    all_targ = [targ for _, targ in dataset]
    norm_tab(all_feats)
    return [(feats, targ) for feats, targ in zip(all_feats, all_targ)]

# shuffle dataset and split it into train and test sets
def prepare_data():
    dataset = norm_dataset(load_dataset())
    random.shuffle(dataset)
    train_size = (4 * len(dataset)) // 5 # train size 80%
    return dataset[:train_size], dataset[train_size:]

# get column of table
def getcol(table, col):
   return [row[col] for row in table] 

# get all columns of table
def getcols(table):
    num_cols = len(table[0])
    return [getcol(table, col) for col in range(num_cols)] 

# MLP functions

# logistic function
def logistic(x):
    return 1 / (1 + math.exp(-x))

# get dot product of 2 lists
def dot(A, B):
    return sum([a * b for a, b in zip(A, B)])

# compute output of a node
# weights are the weights connecting this node to nodes of the previous layer
def node_out(inputs, weights):
    return logistic(dot(inputs, weights))

# compute output for a whole layer using inputs from the previous layer
# and the weight table for the previous layer and this layer
def layer_out(inputs, weights):
    return [node_out(inputs, col) for col in getcols(weights)] # a column stores the upstream weights of a node

# forward propagate inputs
# returns computed activations of the entire net, not just the output layer
def forw_prop(inputs):
    outputs = [inputs]
    for layer in range(1, len(LAYERS)): 
        # compute layer using outputs and weights from previous layer
        outputs.append(layer_out(outputs[layer-1], weights[layer-1]))
    return outputs

# compute output layer gradients
def out_grads(outputs, targets):
    return [(o - t) * o * (1 - o) for o, t in zip(outputs, targets)]

# compute inner layer gradients
# outputs are outputs of the layer
# weights should be the weight table for this and the next layer
# grads should be gradients of next layer
def in_grads(outputs, weights, grads):
    return [dot(row, grads) * o * (1 - o) for o, row in zip(outputs, weights)]

# update weight tables using backprop
def change_weights(outputs, targets):
    grads = None # keeps track of the last computed gradients i.e. layer+1 in below loop
    out_layer = len(LAYERS)-1 # output layer
    # goes from output-1 to layer 0. at each iteration, compute gradients of layer+1 and update the weight table for layer and layer+1 
    for layer in reversed(range(out_layer)):
        if (layer == out_layer-1): # how grads of layer+1 are computed depends on whether it is an output or inner layer 
            grads = out_grads(outputs[-1], targets)
        else:
            grads = in_grads(outputs[layer+1], weights[layer+1], grads) # if inner layer, compute new grads from old
        for i in range(LAYERS[layer]):
            for j in range(LAYERS[layer+1]):
                weights[layer][i][j] -= LEARN_RATE * outputs[layer][i] * grads[j]

# fit a single example
def fit(inputs, target):
    outputs = forw_prop(inputs)
    print(f'f {target} {forw_prop(inputs)[1:]}')
    change_weights(outputs, [target])

# fit a list of examples
def train(data):
    for feats, targ in data:
        fit(feats, targ)
        
# predict class of an example
def predict(inputs):
    outputs = forw_prop(inputs)
    print(f'p {outputs[1:]}')
    return round(outputs[-1][0])

# make predictions on data and return number of correct ones
def test(data):
    num_correct = 0
    for feats, target in data:
        if (predict(feats) == target):
            num_correct += 1
    return num_correct

# save weights to file
def save_weights():
    with open('weights', 'wb') as f:
        pickle.dump(weights, f)

# load weights from file
def load_weights():
    with open('weights', 'rb') as f:
        weights = pickle.load(f)

# Entrypoint

# train and test
train_set, test_set = prepare_data()
# load_weights()
for _ in range(LEARN_ITERS):
    train(train_set)
save_weights()
num_correct = test(test_set)

accuracy = num_correct / len(test_set) * 100
print(f'Accuracy: {round(accuracy, 2)}%')