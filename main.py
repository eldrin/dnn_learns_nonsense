#! /usr/local/bin/python3
from __future__ import print_function
import time

import numpy as np
import lasagne
from lasagne import layers as L
from lasagne.objectives import squared_error
from lasagne.updates import sgd,adagrad,adam,apply_momentum
from lasagne.nonlinearities import rectify,identity
from lasagne.regularization import regularize_layer_params, l2, l1

import theano
from theano import tensor as T

floatX = theano.config.floatX

N_SAMPLES = 10000
N_TRAIN = 10
SZ_BATCH = 5

N_DIM = 5
N_HID = 5
N_ITER = 5000

PATIENCE = 100

OPTIMIZER = sgd

LR = 0.0000001
LR_SHARED = theano.shared(np.array(LR,dtype=floatX))
LR_DECAY = 0.8
LR_DECAY_RATE = 100

def model_function(a,b):
	'''
	function that we want to train through NN

	(assume both input vector's lengths are same)

	y = (a + bIv)w

	'''
	try:
		assert b.shape[1]==a.shape[1]
	except Exception as e:
		print('Two inputs have different lengths!')

	ndim = a.shape[1]
	Iv = np.fliplr(np.eye(ndim)) # get flipping matrix
	w = np.array([10**i for i in range(ndim)]) # weight matrix

	# model function
	y = (a + b.dot(Iv)).dot(w)[:,None]

	return y


def create_dataset(n_samples,n_train,sz_input_vec=5):
	'''
	create samples
	'''

	A = np.random.randint(0,9,(n_samples,sz_input_vec))
	B = np.random.randint(0,9,(n_samples,sz_input_vec))

	y = model_function(A, B)

	train_set = (A[:n_train],B[:n_train],y[:n_train])
	test_set = (A[n_train:],B[n_train:],y[n_train:])

	return train_set,test_set


def build_model(n_input,n_hidden,optimizer=adagrad,
				l2_weight=1e-4,l1_weight=1e-2):
	'''
	build NN model to estimating model function
	'''
	global LR

	input_A = L.InputLayer((None,n_input),name='A')
	layer_A = L.DenseLayer(input_A,n_hidden,b=None,nonlinearity=identity)

	input_B = L.InputLayer((None,n_input),name='B')
	layer_B = L.DenseLayer(input_B,n_hidden,b=None,nonlinearity=identity)

	merge_layer = L.ElemwiseSumLayer((layer_A,layer_B))

	output_layer = L.DenseLayer(merge_layer,1,b=None,nonlinearity=identity) # output is scalar

	x1 = T.matrix('x1')
	x2 = T.matrix('x2')
	y = T.matrix('y')

	out = L.get_output(output_layer,{input_A:x1,input_B:x2})
	params = L.get_all_params(output_layer)
	loss = T.mean(squared_error(out,y))

	# add l1 penalty
	l1_penalty = regularize_layer_params([layer_A,layer_B,output_layer],l1)

	# add l2 penalty
	l2_penalty = regularize_layer_params([layer_A,layer_B,output_layer],l2)

	# get loss + penalties
	loss = loss + l1_penalty*l1_weight + l2_penalty*l2_weight

	updates_sgd = optimizer(loss,params,learning_rate=LR)
	updates = apply_momentum(updates_sgd,params,momentum=0.9)
	# updates = optimizer(loss,params,learning_rate=LR)

	f_train = theano.function([x1,x2,y],loss,updates=updates)
	f_test = theano.function([x1,x2,y],loss)
	f_out = theano.function([x1,x2],out)

	return f_train,f_test,f_out,output_layer


def fit_model(dataset,f_train,f_test,sz_batch=128,
				n_iter=100,debug_print_rate=10,patience=10):
	'''
	train to optimize NN find model function
	'''
	global LR_SHARED,LR_DECAY,LR_DECAY_RATE

	train_set,test_set = dataset
	A,B,y = train_set
	At,Bt,yt = test_set

	if sz_batch > A.shape[0]:
		sz_batch = A.shape[0]

	print('Start Training...')
	
	if A.shape[0]%sz_batch==0:
		num_batches_train = int(A.shape[0]/sz_batch)
	else:
		num_batches_train = int(A.shape[0]/sz_batch)+1

	if At.shape[0]%sz_batch==0:
		num_batches_test = int(At.shape[0]/sz_batch)
	else:
		num_batches_test = int(At.shape[0]/sz_batch)+1

	train_losses = []
	valid_losses = []
	counter = 0
	try:
		for epoch in range(n_iter):
			epoch_st = time.time()

			# decaying learning rate
			if (epoch+1) % LR_DECAY_RATE == 0:
				LR_SHARED.set_value(
					np.asarray(
						LR_SHARED.get_value()*LR_DECAY,
						dtype=floatX
					)
				)

			tloss = 0
			N = 0
			for batch in range(num_batches_train):

				batch_st = time.time()
				
				# get batch
				slc = slice(sz_batch*batch,sz_batch*(batch+1))
				X1_batch, X2_batch, y_batch = A[slc],B[slc],y[slc]

				# number of batch rows
				M = X1_batch.shape[0]
				N += M

				train_res = f_train(X1_batch,X2_batch,y_batch)
				tloss += train_res

				batch_end = time.time()
				if batch%debug_print_rate==0:
					print("Batch {:d}/{:d} - {:.2f}s - tloss: {:.4f}".format(\
							batch+1, num_batches_train, 
							batch_end-batch_st, 
							float(train_res)/M),end='\r')

			train_loss = tloss / float(N)
			train_losses.append(train_loss)

			vloss = 0
			Nt = 0
			for batch in range(num_batches_test):

				# get batch
				slc = slice(sz_batch*batch,sz_batch*(batch+1))
				Xt1_batch, Xt2_batch, yt_batch = At[slc],Bt[slc],yt[slc]

				# number of batch rows
				M = Xt1_batch.shape[0]
				Nt += M

				valid_res = f_test(Xt1_batch,Xt2_batch,yt_batch)
				vloss += valid_res

			valid_loss = vloss / float(Nt)
			valid_losses.append(valid_loss)

			epoch_end = time.time()
			print("Epoch {}/{} - {:.1f}s - tloss: {:.4f} - vloss: {:.4f}      ".format(\
					epoch+1, n_iter, epoch_end-epoch_st , train_loss, valid_loss))

			if len(train_losses)>1:
				if train_losses[-1] > train_losses[-2]:
					if counter < patience:
						counter += 1
					else:
						print("[Early Stopping] No more patience! I'm out!")
						break

	except KeyboardInterrupt:
		print('Interrupted by user.')

	return train_losses


def get_weights(layers):
	'''
	extract weight from NN
	'''
	weights = L.get_all_param_values(layers)
	return weights


if __name__ == '__main__':

	dset = create_dataset(N_SAMPLES, N_TRAIN, N_DIM)
	f_train,f_test,f_out,layers = build_model(N_DIM,N_HID,optimizer=OPTIMIZER)
	loss_curve = fit_model(dset,f_train,f_test,sz_batch=SZ_BATCH,n_iter=N_ITER,patience=PATIENCE)

	w = get_weights(layers)
