import numpy as np
from numpy.linalg import inv, qr, eig, norm
import math
from math import isclose, sqrt

import tensorflow as tf
tf.reset_default_graph()

###############################################################################
######################## MNIST DATA ###########################################
###############################################################################
import input_MNIST_data
from input_MNIST_data import shuffle_data
data = input_MNIST_data.read_data_sets("./data/", one_hot=True)
X_train, y_train = shuffle_data(data)
# input and output shape
n_input   = data.train.images.shape[1]  # here MNIST data input (28,28)
n_classes = data.train.labels.shape[1]  # here MNIST (0-9 digits)

X_train_multi = []
y_train_multi = []
minibatch = 1024
###############################################################################
######################## LBFGS PARAMS #########################################
###############################################################################
m = 10
S = np.array([[]])
Y = np.array([[]])
gamma = 0.2
# GLOBAL VARIABLES - MATRICES
P_ll = [] # P_parallel 
g_ll = []	# g_Parallel
g_NL_norm = 0
Lambda_1 = []

g = []
###############################################################################
######################## LeNet-5 Network Architecture #########################
###############################################################################
# number of weights and bias in each layer
n_W = {}
dim_w = {}

# network architecture hyper parameters
input_shape = [-1,28,28,1]
W0 = 28
H0 = 28

# Layer 1 -- conv
D1 = 1; F1 = 5; K1 = 20; S1 = 1
W1 = (W0 - F1) // S1 + 1
H1 = (H0 - F1) // S1 + 1
conv1_dim = [F1, F1, D1, K1]
conv1_strides = [1,S1,S1,1] 
n_W['1_w_conv'] = F1 * F1 * D1 * K1
n_W['1_b_conv'] = K1 
dim_w['1_w_conv'] = [F1, F1, D1, K1]
dim_w['1_b_conv'] = [K1]

# Layer 2 -- max pool
D2 = K1; F2 = 2; K2 = D2; S2 = 2
W2 = (W1 - F2) // S2 + 1
H2 = (H1 - F2) // S2 + 1
layer2_ksize = [1,F2,F2,1]
layer2_strides = [1,S2,S2,1]

# Layer 3 -- conv
D3 = K2; F3 = 5; K3 = 50; S3 = 1
W3 = (W2 - F3) // S3 + 1
H3 = (H2 - F3) // S3 + 1
conv2_dim = [F3, F3, D3, K3]
conv2_strides = [1,S3,S3,1] 
n_W['2_w_conv'] = F3 * F3 * D3 * K3
n_W['2_b_conv'] = K3 
dim_w['2_w_conv'] = [F3, F3, D3, K3]
dim_w['2_b_conv'] = [K3]

# Layer 4 -- max pool
D4 = K3; F4 = 2; K4 = D4; S4 = 2
W4 = (W3 - F4) // S4 + 1
H4 = (H3 - F4) // S4 + 1
layer4_ksize = [1,F4,F4,1]
layer4_strides = [1,S4,S4,1]


# Layer 5 -- fully connected
n_in_fc = W4 * H4 * D4
n_hidden = 500
fc_dim = [n_in_fc,n_hidden]
n_W['3_w_fc'] = n_in_fc * n_hidden
n_W['3_b_fc'] = n_hidden
dim_w['3_w_fc'] = [n_in_fc,n_hidden]
dim_w['3_b_fc'] = [n_hidden]
# Layer 6 -- output
n_in_out = n_hidden
n_W['4_w_fc'] = n_hidden * n_classes
n_W['4_b_fc'] = n_classes
dim_w['4_w_fc'] = [n_hidden,n_classes]
dim_w['4_b_fc'] = [n_classes]


for key, value in n_W.items():
	n_W[key] = int(value)

###############################################################################
######################## f(x;w) ###############################################
###############################################################################
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

w_initializer = tf.contrib.layers.xavier_initializer()

w_tf = {}
for key, _ in dim_w.items():
	w_tf[key] = tf.get_variable(key, shape=dim_w[key], initializer=w_initializer)

def lenet5_model(x,_w):
	# Reshape input to a 4D tensor 
	x = tf.reshape(x, shape = input_shape)
	# LAYER 1 -- Convolution Layer
	conv1 = tf.nn.relu(tf.nn.conv2d(input = x, 
									filter =_w['1_w_conv'],
									strides = [1,S1,S1,1],
									padding = 'VALID') + _w['1_b_conv'])
	# Layer 2 -- max pool
	conv1 = tf.nn.max_pool(	value = conv1, 
							ksize = [1, F2, F2, 1], 
							strides = [1, S2, S2, 1], 
							padding = 'VALID')

	# LAYER 3 -- Convolution Layer
	conv2 = tf.nn.relu(tf.nn.conv2d(input = conv1, 
									filter =_w['2_w_conv'],
									strides = [1,S3,S3,1],
									padding = 'VALID') + _w['2_b_conv'])
	# Layer 4 -- max pool
	conv2 = tf.nn.max_pool(	value = conv2 , 
							ksize = [1, F4, F4, 1], 
							strides = [1, S4, S4, 1], 
							padding = 'VALID')
	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer
	fc = tf.contrib.layers.flatten(conv2)
	fc = tf.nn.relu(tf.matmul(fc, _w['3_w_fc']) + _w['3_b_fc'])
	# fc = tf.nn.dropout(fc, dropout_rate)

	y_ = tf.matmul(fc, _w['4_w_fc']) + _w['4_b_fc']
	return y_

# Construct model
model = lenet5_model
y_ = model(x,w_tf)

# Softmax loss
loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_))

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###############################################################################
######################## TF GRADINETS #########################################
###############################################################################
grad_w_tf = {}
for layer, _ in w_tf.items():
	grad_w_tf[layer] = tf.gradients(xs=w_tf[layer], ys=loss)

###############################################################################
######################## TF Auxilary variables ################################
###############################################################################
aux_w = {}
for layer, _ in w_tf.items():
	name = layer + 'aux_w_'
	aux_w[layer] = tf.get_variable(name=name, 
					shape=w_tf[layer].get_shape(), initializer=w_initializer)

aux_w_placeholder = {}
for layer, _ in w_tf.items():
	aux_w_placeholder[layer] = tf.placeholder(dtype="float",
										shape=w_tf[layer].get_shape())
aux_w_init = {}
for layer, _ in w_tf.items():
	aux_w_init[layer] = aux_w[layer].assign(aux_w_placeholder[layer])

aux_output = model(x,aux_w)
aux_loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = aux_output))
aux_grad_w = {}
for layer, _ in w_tf.items():
	aux_grad_w[layer] = tf.gradients(xs=aux_w[layer], ys=aux_loss)

update_w = {}
update_w_placeholder = {}
for layer, _ in w_tf.items():
	update_w_placeholder[layer] = tf.placeholder(dtype="float",
										shape=w_tf[layer].get_shape())
for layer, _ in w_tf.items():
	update_w[layer] = w_tf[layer].assign(update_w_placeholder[layer])

###############################################################################
###############################################################################
saver = tf.train.Saver()
init = tf.global_variables_initializer()
###############################################################################
###############################################################################


def quad_model():
	return 0

def phi_bar_func(sigma,delta):
	# phi(sigma) = 1 / v(sigma) - 1 / delta	
	u = sum( (g_ll ** 2) / ((Lambda_1 + sigma) ** 2) ) + \
									(g_NL_norm ** 2) / ( (gamma + sigma) ** 2)
	v = sqrt(u) 

	phi = 1 / v - 1 / delta
	return phi

def phi_bar_prime_func(sigma):
	u = sum( g_ll ** 2 / (Lambda_1 + sigma) ** 2 ) + \
										g_NL_norm ** 2 / (gamma + sigma) ** 2

	u_prime = sum( g_ll ** 2 / (Lambda_1 + sigma) ** 3 ) + \
										g_NL_norm ** 2 / (gamma + sigma) ** 3
	phi_bar_prime = u ** (-3/2) * u_prime

	return phi_bar_prime


def solve_newton_equation_to_find_sigma(delta):
	# tolerance
	tol = 1E-4
	sigma = max( 0, -Lambda_1[1] )
	if phi_bar_func( sigma,delta,Lambda_1,g_ll,g_NL_norm ) < 0:
		sigma_hat = max( abs( g_ll ) / delta - Lambda_1 )
		sigma = max( 0, sigma_hat)
		while( abs( phi_bar_func(sigma,delta,Lambda_1,g_ll,g_NL_norm) ) > tol ):
			phi_bar = phi_bar_func(sigma,delta,Lambda_1,g_ll,g_NL_norm)
			phi_bar_prime = phi_bar_prime_func(sigma,Lambda_1,g_ll,g_NL_norm)
			sigma = sigma - phi_bar / phi_bar_prime
		sigma_star = sigma
	elif Lambda_1[1] < 0:
		sigma_star = - Lambda_1[1]
	else:
		sigma_star = 0

	return sigma_star 


def lbfgs_trust_region_subproblem_solver(delta, g):
	# dimension of w
	n = sum(n_W.values())

	Psi = np.concatenate( (gamma*S, Y) ,axis=1)
	
	S_T_Y = S.T @ Y
	L = np.tril(S_T_Y,k=-1)
	U = np.tril(S_T_Y.T,k=-1).T
	D = np.diag( np.diag(S_T_Y) )

	M = - inv( np.block([ 	[gamma * S.T @ S ,	L],
							[     L.T,		   -D] 
			]) )

	Q, R = qr(Psi, mode='reduced')
	eigen_values, eigen_vectors = eig( R @ M @ R.T )

	# sorted eigen values
	idx = eigen_values.argsort()
	eigen_values_sorted = eigen_values[idx]
	eigen_vectors_sorted = eigen_vectors[:,idx]

	Lambda_hat = eigen_values_sorted
	V = eigen_vectors_sorted

	Lambda_1 = gamma + Lambda_hat
	#Lambda_2 = gamma * np.ones( n-len(Lambda_hat) )
	#B_diag = np.concatenate( (Lambda_1, Lambda_2),axis=0 )

	P_ll = Psi @ inv(R) @ V # P_parallel 
	g_ll = P_ll @ g	# g_Parallel
	g_NL_norm = sqrt ( norm(g) ** 2 - norm(g_ll) ** 2 )

	sigma = 0
	phi = phi_bar_func(sigma,delta)

	if phi >= 0:
		sigma_star = 0
		tau_star = gamma
	else:
		sigma_star = solve_newton_equation_to_find_sigma(delta)
		tau_star = gamma + sigma_star

	p_star = - 1 / tau_star * \
			( g - Psi @ inv( tau_star * inv(M) + Psi.T @ Psi ) @ (Psi.T @ g) )

	return p_star

def eval_reduction_ratio(sess,g,p):
	new_f = eval_aux_loss(sess,p)
	current_f = eval_loss(sess)

	ared = current_f - new_f

	if S.size is not 0:
		p_ll = P_ll @ p
		p_NL_norm = sqrt ( norm(p) ** 2 - norm(p_ll) ** 2 )
		p_T_B_p = sum( Lambda_1 * p_ll ** 2)  + gamma * p_NL_norm ** 2
	else:
		p_T_B_p = gamma * p @ p
	
	pred =  - (g @ p  + 1/2 * p_T_B_p)
	rho = ared / pred
	
	return rho

def eval_y(sess):
	new_g = eval_aux_gradient_vec(sess)
	old_g = g
	new_y = new_g - old_g

	return new_y

def enqueue(Z,new_val):
	if len(Z) == 0:
		Z = new_val
		return Z
	Z = np.concatenate( (Z,new_val), axis=1)
	return Z
		
def dequeue(Z):
	return np.delete(Z, obj=0, axis=1)

def update_S_Y(new_s_val,new_y_val):
	num_columns_S = S.shape[1]
	num_columns_Y = Y.shape[1]
	assert num_columns_S is num_columns_Y, "dimention of S and Y doesn't match"
	if num_columns_S < m:
		S = enqueue(S,new_s_val)
		Y = enqueue(Y,new_y_val)
		return
	else:
		S = dequeue(S)
		S = enqueue(S,new_s_val)
		Y = dequeue(Y)
		Y = enqueue(Y,new_y_val)
		return 


def dict_of_weight_matrices_to_single_linear_vec(x_dict):
	x_vec = np.array([])
	for key in sorted(w_tf.keys()):
		matrix = x_dict[key]
		x_vec = np.append(x_vec,matrix.flatten())	
	#x_vec = x_vec.reshape(-1 , 1)
	return x_vec

def linear_vec_to_dict_of_weight_matrices(x_vec):
	x_dict = {}
	id_start = 0
	id_end   = 0
	for key in sorted(w_tf.keys()):
		id_end = id_start + n_W[key]
		vector = x_vec[id_start:id_end]
		matrix = vector.reshape(dim_w[key])
		x_dict[key] = matrix
		id_start = id_end
	return x_dict

def compute_multibatch_tensor(sess,tensor_tf,feed_dict,X_train,y_train):
	total = 0
	num_minibatches_here = X_train.shape[0] // minibatch
	for j in range(num_minibatches_here):
		index_minibatch = j % num_minibatches_here
		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = X_train[start_index:end_index]
		y_batch = y_train[start_index:end_index]
		feed_dict.update({	x: X_batch,
							y: y_batch})

		value = sess.run(tensor_tf, feed_dict=feed_dict)
		total = total + value

	total = total * 1 / num_minibatches_here	
	return total

def compute_multibatch_gradient(sess,grad_tf,feed_dict,X_train,y_train):
	gw = {}
	num_minibatches_here = X_train.shape[0] // minibatch
	for j in range(num_minibatches_here):
		index_minibatch = j % num_minibatches_here
		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = X_train[start_index:end_index]
		y_batch = y_train[start_index:end_index]
		feed_dict.update({	x: X_batch,
							y: y_batch})

		gw_list = sess.run(grad_tf, feed_dict=feed_dict)
		if j == 0:		
			for layer, _ in w_tf.items():
				gw[layer] = gw_list[layer][0]
		else:
			for layer, _ in w_tf.items():
				gw[layer] = gw[layer] + gw_list[layer][0]

	for layer, _ in w_tf.items():
		gw[layer] = gw[layer] * 1 / num_minibatches_here	
	return gw

def eval_gradient_vec(sess):
	"""returns gradient, here only for mode='robust-multi-batch' 
	I should modify to consider all other cases"""
	feed_dict = {}
	g_dict = compute_multibatch_gradient(sess,grad_w_tf,feed_dict,
												X_train_multi,y_train_multi)
	g_vec = dict_of_weight_matrices_to_single_linear_vec(g_dict)
	return g_vec	

def eval_accuracy(sess,k,mode='robust-multi-batch'):
	feed_dict = {}
	accuracy_val = compute_multibatch_tensor(sess,accuracy,feed_dict,
												X_train_multi,y_train_multi)
	return accuracy_val

def eval_w_dict(sess):
	w_dict = sess.run(w_tf)
	return w_dict

def update_weights(sess,p_vec):
	w_dict = eval_w_dict(sess)
	p_dict = linear_vec_to_dict_of_weight_matrices(p_vec)
	feed_dict = {}
	for key,_ in w_tf.items():
		feed_dict.update({update_w_placeholder[key]: w_dict[key]+p_dict[key] })
	sess.run(update_w, feed_dict=feed_dict)
	return

def eval_aux_loss(sess,p_vec):
	w_dict = eval_w_dict(sess)
	p_dict = linear_vec_to_dict_of_weight_matrices(p_vec)
	feed_dict = {}
	for key,_ in w_tf.items():
		feed_dict.update({aux_w_placeholder[key]: w_dict[key]+p_dict[key] })
	sess.run(aux_w_init,feed_dict=feed_dict)
	feed_dict = {}
	loss_new = compute_multibatch_tensor(sess,aux_loss,feed_dict,
											X_train_multi,y_train_multi)
	return loss_new

def eval_loss(sess):
	feed_dict = {}
	loss_val = compute_multibatch_tensor(sess,loss,feed_dict,
											X_train_multi,y_train_multi)
	return loss_val

def eval_aux_gradient_vec(sess):
	feed_dict = {}
	aux_g_dict = compute_multibatch_gradient(sess,aux_grad_w,feed_dict,
												X_train_multi,y_train_multi)
	aux_g_vec = dict_of_weight_matrices_to_single_linear_vec(aux_g_dict)
	return aux_g_vec	

###############################################################################
######################## TRUST REGION ALGORITHM ###############################
###############################################################################

#--------- LOOP PARAMS ------------
delta_hat = 20 # upper bound for trust region radius
max_num_iter = 20 # max bunmber of trust region iterations
delta = np.zeros(max_num_iter+1)
delta[0] = delta_hat * 0.75
rho = np.zeros(max_num_iter) # true reduction / predicted reduction ratio
eta = 1/4 * 0.9 # eta \in [0,1/4)
new_iteration = True
flip_batch = 0
with tf.Session() as sess:
	sess.run(init)

	#-------- main loop ----------
	for k in range(max_num_iter):
		
		# batching -- take this to a function
		if new_iteration:
			flip_batch = flip_batch + 1
			start_index = (flip_batch%2)   * X_train.shape[0] // 3
			end_index = (flip_batch%2 + 2) * X_train.shape[0] // 3
			X_train_multi = X_train[start_index:end_index]
			y_train_multi = y_train[start_index:end_index]

		g = eval_gradient_vec(sess)		
		
		if S.size == 0:
			p = -gamma * g
		else:
			p = lbfgs_trust_region_subproblem_solver(delta[k], g)
		
		rho[k] = eval_reduction_ratio(sess, g, p)
		if rho[k] < 1/4:
			delta[k+1] = 1/4 * delta[k]
		else:
			if rho[k] > 3/4 and isclose( norm2(p), delta[k] ):
				delta[k+1] = min(2*delta[k], delta_hat)
			else:
				delta[k+1] = delta[k]

		if rho[k] > eta:
			update_weights(sess,p)
			y = eval_y(sess)
			s = p
			update_S_Y(s,y)
			gamma = (y.T @ y) / (s.T @ y)
			new_iteration = True
			print('weight is update')
		else:
			new_iteration = False
			continue






