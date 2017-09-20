import tensorflow as tf
import numpy as np

def weight(shape):
	return tf.get_variable('weight', shape, initializer=tf.contrib.layers.xavier_initializer())

def bias(shape, value=0.1):
	return tf.get_variable('bias', shape, initializer=tf.constant_initializer(value))


def Fcnn(x,insize,outsize,name,activation=None,nobias=False):
	with tf.variable_scope(name):
		if nobias:
			print('No biased fully connected layer is used!')
			W = weight([insize,outsize])
			tf.summary.histogram(name+'/weight',W)
			if activation==None:
				return tf.matmul(x,W)
			return activation(tf.matmul(x,W))
		else:
			W = weight([insize,outsize])
			b = bias([outsize])
			tf.summary.histogram(name+'/weight',W)
			tf.summary.histogram(name+'/bias',b)
			if activation==None:
				return tf.matmul(x,W)+b
			return activation(tf.matmul(x,W)+b)

def conv2D(x,kernel_size,outchn,name,stride=1,pad='SAME', usebias=True):
	print('Conv_bias:',usebias)
	# with tf.variable_scope(name):
	# if isinstance(size,list):
	# 	kernel = size
	# else:
	kernel = [kernel_size, kernel_size]
	z = tf.layers.conv2d(x, outchn, kernel, strides=(stride, stride), padding=pad,\
		kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),\
		use_bias=usebias,\
		bias_initializer=tf.constant_initializer(0.1),name=name)
	# print ('z:', z.get_shape())
	return z

def maxpooling(x,size,stride,name,pad='SAME'):
	with tf.variable_scope(name):
		return tf.nn.max_pool(x,ksize=[1,size,size,1],strides=[1,stride,stride,1],padding=pad)

def avgpooling(x,size,stride,name,pad='SAME'):
	with tf.variable_scope(name):
		return tf.nn.avg_pool(x,ksize=[1,size,size,1],strides=[1,stride,stride,1],padding=pad)


def MFM(x,name):
	with tf.variable_scope(name):
		#shape is in format [batchsize, x, y, channel]
		# shape = tf.shape(x)
		shape = x.get_shape().as_list()
		res = tf.reshape(x,[-1,shape[1],shape[2],2,shape[-1]//2])
		# x2 = tf.reshape(x,[-1,2,shape[1]//2, shape[2], shape[3]])
		res = tf.reduce_max(res,axis=[3])
		# x2 = tf.reduce_max(x2,axis=[1])
		# x3 = tf.reshape(x2,[-1,int(x2.get_shape()[3]), int(x2.get_shape()[2]), int(x2.get_shape()[1])])
		return res

def MFMfc(x,half,name):
	with tf.variable_scope(name):
		shape = x.get_shape().as_list()
		# print('fcshape:',shape)
		res = tf.reduce_max(tf.reshape(x,[-1,2,shape[-1]//2]),reduction_indices=[1])
	return res

def batch_norm(inp,name,training=True):
	print('BN training:',training)
	return tf.layers.batch_normalization(inp,training=training,name=name)

def L2_norm(inp, dim):
	print ('L2 normlization...')
	return tf.nn.l2_norm(inp, dim)

def lrelu(x,name,leaky=0.2):
	return tf.maximum(x,x*leaky,name=name)

def relu(inp,name):
	return tf.nn.relu(inp,name=name)

def tanh(inp,name):
	return tf.tanh(inp,name=name)

def elu(inp,name):
	return tf.nn.elu(inp,name=name)

def sparse_softmax_cross_entropy(inp,lab,name):
	with tf.name_scope(name):
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab,logits=inp))
		return loss

def sigmoid(inp,name):
	return tf.sigmoid(inp,name=name)

def resize_nn(inp,size,name):
	with tf.name_scope(name):
		return tf.image.resize_nearest_neighbor(inp,size=(int(size),int(size)))

def accuracy(pred,y,name):
	with tf.variable_scope(name):
		# a = tf.cast(tf.argmax(pred,1),tf.int64)
		# b = tf.cast(tf.argmax(y, 1),tf.int64)
		# c = tf.argmax(pred,1)
		correct = tf.equal(tf.cast(tf.argmax(pred,1),tf.int64),tf.cast(y,tf.int64))
		# correct = tf.equal(tf.cast(tf.argmax(pred,1),tf.int64),tf.cast(tf.argmax(y, 1),tf.int64))
		acc = tf.reduce_mean(tf.cast(correct,tf.float32))
		# #acc = tf.cast(correct,tf.float32)
		return acc

def dropout(inp, keep_prob):
	return tf.nn.dropout(inp,keep_prob)

# test
# test accuracy
# f_log = open('log.txt', 'w')
# y = [[1,0,0],
# [0,1,0],
# [0,0,1]] 

# pred = [[0.92, 0.4, 0.4],
# [0.2,0.6,0.2],
# [0.2,0.4,0.2]]

# acc, correct, a, b, c = accuracy(pred,y,'accuracy')
# f_log.write('acc shape: ' + str(acc.get_shape()) + '\n')
# f_log.write('correct shape: ' + str(correct.get_shape()) + '\n')
# f_log.write('a shape: ' + str(a.get_shape()) + '\n')
# f_log.write('b shape: ' + str(b.get_shape()) + '\n')
# f_log.write('c shape: ' + str(c.get_shape()) + '\n')

# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
# 	sess.run(init_op)
# 	f_log.write('acc: ' + str(sess.run(acc)) + '\n')
# 	f_log.write('correct: ' + str(sess.run(correct)) + '\n')
# 	f_log.write('a: ' + str(sess.run(a)) + '\n')
# 	f_log.write('b: ' + str(sess.run(b)) + '\n')
# 	f_log.write('c: ' + str(sess.run(c)) + '\n')
# f_log.close()

# test batch_norm
# f_log = open('log.txt', 'w')
# x1 = tf.get_variable("x1", [1, 2, 3, 4], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

# f_log.write('x1 shape: ' + str(x1.get_shape()) + '\n')

# x2 = batch_norm(x1, 'batch_norm')
# f_log.write('x2 shape: ' + str(x2.get_shape()) + '\n')

# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
# 	sess.run(init_op)
# 	f_log.write('x1: ' + str(sess.run(x1)) + '\n')
# 	f_log.write('x2: ' + str(sess.run(x2)) + '\n')
# f_log.close()


# test bias and weight
# f_log = open('log.txt', 'w')
# b1 = tf.get_variable("b1", [1, 2, 3], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
# w1 = tf.get_variable("w1", [3, 2, 1])

# f_log.write('b1 shape: ' + str(b1.get_shape()) + '\n')
# f_log.write('w1 shape: ' + str(w1.get_shape()) + '\n')

# B1 = bias([b1.get_shape()[0], b1.get_shape()[1], b1.get_shape()[2]])
# # W1 = weight([w1.get_shape()[0], w1.get_shape()[1], w1.get_shape()[2]])
# W1 = weight(w1.get_shape())

# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
# 	sess.run(init_op)
# 	f_log.write('b1: ' + str(sess.run(b1)) + '\n')
# 	f_log.write('B1: ' + str(sess.run(B1)) + '\n')

# 	f_log.write('w1: ' + str(sess.run(w1)) + '\n')
# 	f_log.write('W1: ' + str(sess.run(W1)) + '\n')
# f_log.close()

# test Fcnn
# f_log = open('log.txt', 'w')
# x1 = tf.get_variable("x1", [1, 2, 3], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

# f_log.write('x1 shape: ' + str(x1.get_shape()) + '\n')
# insize = int(x1.get_shape()[0]*x1.get_shape()[1]*x1.get_shape()[2])
# outsize = 3

# x1_reshape = tf.reshape(x1, [-1, int(insize)])
# y1 = Fcnn(x1_reshape,insize,outsize,'fc1')
# y2 = Fcnn(x1_reshape,insize,outsize,'fc2',activation =None, nobias=True)

# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
# 	sess.run(init_op)
# 	f_log.write('x1: ' + str(sess.run(x1)) + '\n')
# 	f_log.write('y1: ' + str(sess.run(y1)) + '\n')
# 	f_log.write('y2: ' + str(sess.run(y2)) + '\n')
# f_log.close()

# test conv2D
# f_log = open('log.txt', 'w')
# x1 = tf.get_variable("x1", [1, 3, 3, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

# f_log.write('x1 shape: ' + str(x1.get_shape()) + '\n')
# kernel_size = 3
# outchn = 10

# x2 = conv2D(x1,kernel_size,outchn,'conv1',stride=1,pad='SAME',activation=None,usebias=True)
# f_log.write('x2 shape: ' + str(x2.get_shape()) + '\n')

# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
# 	sess.run(init_op)
# 	f_log.write('x1: ' + str(sess.run(x1)) + '\n')
# 	f_log.write('x2: ' + str(sess.run(x1)) + '\n')
# f_log.close()

# test maxpooling avgpooling
# f_log = open('log.txt', 'w')
# x1 = tf.get_variable("x1", [1, 3, 3, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

# f_log.write('x1 shape: ' + str(x1.get_shape()) + '\n')
# size = 3
# stride = 1

# x2 = maxpooling(x1,size,stride,'maxpooling1',pad='SAME')
# x3 = avgpooling(x1,size,stride,'avgpooling1',pad='SAME')
# f_log.write('x2 shape: ' + str(x2.get_shape()) + '\n')
# f_log.write('x3 shape: ' + str(x3.get_shape()) + '\n')

# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
# 	sess.run(init_op)
# 	f_log.write('x1: ' + str(sess.run(x1)) + '\n')
# 	f_log.write('x2: ' + str(sess.run(x2)) + '\n')
# 	f_log.write('x3: ' + str(sess.run(x3)) + '\n')
# f_log.close()

# test MFM
# f_log = open('log.txt', 'w')
# x1 = tf.get_variable("x1", [1, 4, 3, 4], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

# f_log.write('x1 shape: ' + str(x1.get_shape()) + '\n')

# x2, y2, y3= MFM(x1,'.', 'mfm1')
# f_log.write('x2 shape: ' + str(x2.get_shape()) + '\n')
# f_log.write('y2 shape: ' + str(y2.get_shape()) + '\n')

# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
# 	sess.run(init_op)
# 	f_log.write('x1: ' + str(sess.run(x1)) + '\n')
# 	f_log.write('x2: ' + str(sess.run(x2)) + '\n')
# 	f_log.write('y2: ' + str(sess.run(y2)) + '\n')
# 	f_log.write('y3: ' + str(sess.run(y3)) + '\n')
# f_log.close()
