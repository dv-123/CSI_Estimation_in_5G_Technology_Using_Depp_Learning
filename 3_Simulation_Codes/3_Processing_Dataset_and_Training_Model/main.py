import numpy as np

# loading the CSI infor for the URLLC
CSI = np.load("HMMSE.npy")
r_CSI = np.load("r_HMMSE.npy")
theta_CSI = np.load("theta_HMMSE.npy")

# Loading the channel information for eMMB
h = np.load("rxSig_h.npy")
r_h = np.load("r.npy")
theta_h = np.load("theta.npy")

# for the model we will be taking the angular domain of the CSI
angular_CSI = theta_CSI.astype(np.float32)

# now doing some data preprocessing

# for the network parameters we require,
# Input --> CSI
# for loss function we require,
# beamforming weights -->
# here we need
# w_opt =  h / ||h||^2
# w_inferred = which is the output vector form the network
# transpose(w_label) = 1/h --> constant --> from the vehicle

# the loss function will be -->
# Loss(w_infered, w_opt) = 1 - (w_inferred*w_label)/(w_opt*w_label)

# vector form for makin w_opt.
# 4 h vectors for 4 receiving antennas
h_1 = np.expand_dims(h[:,0], axis=1)
h_2 = np.expand_dims(h[:,1], axis=1)
h_3 = np.expand_dims(h[:,2], axis=1)
h_4 = np.expand_dims(h[:,3], axis=1)

w_opt_1 = np.divide(h_1,np.square((np.matmul(np.transpose(h_1),h_1))))
w_opt_2 = np.divide(h_2,np.square((np.matmul(np.transpose(h_2),h_2))))
w_opt_3 = np.divide(h_3,np.square((np.matmul(np.transpose(h_3),h_3))))
w_opt_4 = np.divide(h_4,np.square((np.matmul(np.transpose(h_4),h_4))))

# angular domain transformation
w_opt_1_angle = np.angle(w_opt_1)
w_opt_2_angle = np.angle(w_opt_2)
w_opt_3_angle = np.angle(w_opt_3)
w_opt_4_angle = np.angle(w_opt_4)

# combining the weights.
w_opt_angle = np.array([w_opt_1_angle,w_opt_2_angle,w_opt_3_angle,w_opt_4_angle], dtype = "float32")
w_opt_angle = np.squeeze(w_opt_angle, axis = 2)
w_opt_angle = np.rollaxis(w_opt_angle, axis = -1)

# now lets set the w_label,
w_label_1 = 1/h
w_lable_angle = np.angle(w_label_1)
# now we state that the mobile eMMBc vehicle is stationary in the cell so taking a constatnt w_label
# used if required otherwise we will be feeding the mobile data from the eMBB network
w_lable_angle_constant = w_lable_angle[1997].astype(np.float32)
# this is taken randomly.

################################### now training the model #######################################
import tensorflow as tf
from matplotlib import pyplot as plt

BATCH_SIZE = 1
learning_rate = 0.0001
epochs = 1000

input_dim = 4           # number of antennas in array --> CSI for 4 antennas
hidden_dim = 128
output_dim = 4
label_dim = 4

# A custom initialization (see Xavier Glorot init)
def xavier_init(shape):
   return tf.random.normal(shape = shape, stddev = 1./tf.sqrt(shape[0]/2.0))

#Define weight and bias dictionaries
weights = {"hidd_1"     : tf.Variable(xavier_init([input_dim, hidden_dim])),
            "hidd_2"    : tf.Variable(xavier_init([hidden_dim, hidden_dim])),
            "hidd_3"    : tf.Variable(xavier_init([hidden_dim, hidden_dim])),
            "output"    : tf.Variable(xavier_init([hidden_dim, output_dim]))}

bias    = {"hidd_1"     : tf.Variable(xavier_init([hidden_dim])),
            "hidd_2"    : tf.Variable(xavier_init([hidden_dim])),
            "hidd_3"    : tf.Variable(xavier_init([hidden_dim])),
            "output"    : tf.Variable(xavier_init([output_dim]))}

# Define Model Network
def model(x):
    hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(x,weights["hidd_1"]), bias["hidd_1"]))
    hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_1,weights["hidd_2"]), bias["hidd_2"]))
    hidden_layer_3 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_2,weights["hidd_3"]), bias["hidd_3"]))
    final_layer = tf.add(tf.matmul(hidden_layer_3, weights["output"]), bias["output"])
    output = tf.nn.softmax(final_layer)
    return output

# Define the placeholders for External input
input = tf.placeholder(tf.float32, shape = [None,  input_dim], name = "input")

# defining the placeholder for required output
w_opt = tf.placeholder(tf.float32, shape = [None, output_dim], name = "w_opt")

# defining the constant
w_label = tf.placeholder(tf.float32, shape = [None, label_dim], name = "w_label")

# Building the Model Network
w_infered = model(input)

# Defining the loss function --> error may occur
Loss_1 = 1 - tf.math.divide(tf.math.multiply(w_infered,w_label),tf.math.multiply(w_opt,w_label))
Loss = tf.math.abs(tf.math.divide(tf.math.reduce_sum(Loss_1),4))

# Defining the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(Loss)

# Initialize Variables
init = tf.global_variables_initializer()

# Start training
sess = tf.Session()
sess.run(init)

L = []

for epoch in range(epochs):
    i = epoch%len(angular_CSI)
    batch_x = angular_CSI[i:i+1]
    batch_y = w_opt_angle[i:i+1]
    batch_z = w_lable_angle[i:i+1]

    _, loss = sess.run([train, Loss], feed_dict = {input: batch_x, w_opt: batch_y, w_label: batch_z})

    L.append(loss)

    print("Epoch_{}: loss = {}".format(epoch, loss))

Loss = np.array(L)
print("Minimum loss: {}".format(np.amin(Loss, axis=-1)))
