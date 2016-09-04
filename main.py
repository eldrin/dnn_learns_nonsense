import numpy as np
import lasagne
from lasagne import layers as L
from lasagne.objectives import squared_error
from lasagne.updates import sgd,apply_momentum

import theano
from theano.config import floatX
from theano import tensor as T

N_SAMPLES = 1000
N_TRAIN = 900

LR = 0.001
LR_SHARED = theano.shared(np.array(LR,dtype=floatX))
LR_DECAY = 0.9
LR_DECAY_RATE = 10

def model_function(a,b):
	'''
	function that we want to train through NN

	(assume both input vector's lengths are same)

	y = (a + bIv)w

	'''
	try:
		assert len(b)==len(a)
	except Exception as e:
		print('Two inputs have different lengths!')

	ndim = len(a)
	Iv = np.fliplr(np.eye(ndim)) # get flipping matrix
	w = np.array([1,10,100,1000,1000]) # weight matrix

	# model function
	y = (a + b.dot(Iv)).dot(w)

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


def build_model(n_input,n_hidden):
	'''
	build NN model to estimating model function
	'''
	global LR


	input_A = L.InputLayer((None,n_input),name='A')
	layer_A = L.DenseLayer(input_A,n_hidden)

	input_B = L.InputLayer((None,n_input),name='B')
	layer_B = L.DenseLayer(input_B,n_hidden)

	merge_layer = L.ElemwiseSumLayer((layer_A,layer_B))

	output_layer = L.DenseLayer(merge_layer,1) # output is scalar

	x1 = T.matrix('x1')
	x2 = T.matrix('x2')
	y = T.vector('y')

	out = L.get_output(output_layer,{input_A:x1,input_B:x2})
	params = L.get_all_params(output_layer)
	loss = T.mean(squared_error(out,y))

	updates_sgd = sgd(loss,params,learning_rate=LR)
	updates = apply_momentum(updates_sgd,params,momentum=0.9)

	f_train = theano.function([x1,x2,y],loss,updates=updates)
	f_test = theano.function([x1,x2],out)

	return f_train,f_test


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
	
	num_batches_train = int(A.shape[0]/sz_batch)+1
	train_losses = []
	train_accs = []
	counter = 0
	gs = []
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
				X_batch, y_batch, _ = self.dbuilder.get_rows()

				# number of batch rows
				M = X_batch.shape[0]
				N += M

				train_res = self._train(X_batch)
				tloss += train_res

				batch_end = time.time()
				if batch%debug_print_rate==0:
					print("Batch {:d}/{:d} - {:.2f}s - tloss: {:.4f}".format(\
							batch+1, num_batches_train, 
							batch_end-batch_st, 
							float(train_res)/M),end='\r')

			train_loss = tloss / float(N)
			train_losses.append(train_loss)

			epoch_end = time.time()
			print("Epoch {}/{} - {:.1f}s - tloss: {:.4f}           ".format(\
					epoch+1, n_iter, epoch_end-epoch_st , train_loss))

			if len(train_losses)>1:
				if train_losses[-1] > train_losses[-2]:
					if counter < patience:
						counter += 1
					else:
						print("[Early Stopping] No more patience! I'm out!")
						break

	except KeyboardInterrupt:
		print('Interrupted by user.')


if __name__ == '__main__':

	global N_SAMPLES,N_TRAIN
	dset = create_dataset(N_SAMPLES, N_TRAIN)
	