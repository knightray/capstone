#
# Build a CNN model based on cifar-10 example
# 
# by ZhangChengKe
# 2017.12.28
# =======================================================

import tensorflow as tf
import define
import numpy as np
import os
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

slim = tf.contrib.slim

class Model(object):
	def __init__(self, is_trainning):
		self.is_trainning = is_trainning
		pass

	def print_all_variables(self, train_only=True):
    	# tvar = tf.trainable_variables() if train_only else tf.all_variables()
		if train_only:
			t_vars = tf.trainable_variables()
			define.log("  [*] printing trainable variables")
		else:
			t_vars = tf.global_variables()
			define.log("  [*] printing global variables")
		for idx, v in enumerate(t_vars):
			define.log("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

	def conv(self, layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=False):
		in_channels = x.get_shape()[-1]
		with tf.variable_scope(layer_name):
			w = tf.get_variable(name='weights',
					trainable=not is_pretrain,
					shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
					initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable(name='biases',
					trainable=not is_pretrain,
					shape=[out_channels],
					initializer=tf.constant_initializer(0.0))
			x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
			x = tf.nn.bias_add(x, b, name='bias_add')
			x = tf.nn.relu(x, name='relu')
			return x

	def pool(self, layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
		if is_max_pool:
			x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
		else:
			x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
		return x

	def fc_layer(self, layer_name, x, out_nodes, is_pretrain = False):
		shape = x.get_shape()
		if len(shape) == 4:
			size = shape[1].value * shape[2].value * shape[3].value
		else:
			size = shape[-1].value

		with tf.variable_scope(layer_name):
			w = tf.get_variable('weights',
					trainable=not is_pretrain,
					shape=[size, out_nodes],
					initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable('biases',
					trainable=not is_pretrain,
					shape=[out_nodes],
					initializer=tf.constant_initializer(0.0))
			flat_x = tf.reshape(x, [-1, size]) # flatten into 1D

			x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
			x = tf.nn.relu(x)
		return x

	def dropout(self, layer_name, x):
		if self.is_trainning:
			keep_prob = define.KEEP_PROB
		else:
			keep_prob = 1

		with tf.variable_scope(layer_name):
			x = tf.nn.dropout(x, keep_prob)
		return x

	def softmax_linear(self, layer_name, x, out_nodes):

		shape = x.get_shape()
		if len(shape) == 4:
			size = shape[1].value * shape[2].value * shape[3].value
		else:
			size = shape[-1].value
		with tf.variable_scope(layer_name) as scope:
			weights = tf.get_variable('softmax_linear',
					shape=[size, out_nodes],
					dtype=tf.float32,
					initializer=tf.contrib.layers.xavier_initializer())
			biases = tf.get_variable('biases',
					shape=[out_nodes],
					dtype=tf.float32,
					initializer=tf.constant_initializer(0.0))
			softmax_linear = tf.add(tf.matmul(x, weights), biases, name='softmax_linear')

		return softmax_linear

	def losses(self, logits, labels):
		with tf.variable_scope('loss') as scope:
			cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
							(logits=logits, labels=labels, name='xentropy_per_example')
			loss = tf.reduce_mean(cross_entropy, name='loss')
			tf.summary.scalar(scope.name+'/loss', loss)
		return loss

	def trainning(self, loss, learning_rate):
		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
			global_step = tf.Variable(0, name='global_step', trainable=False)
			train_op = optimizer.minimize(loss, global_step= global_step)
		return train_op

	def evaluation(self, logits, labels):
		with tf.variable_scope('accuracy') as scope:
			correct = tf.nn.in_top_k(logits, labels, 1)
			correct = tf.cast(correct, tf.float16)
			accuracy = tf.reduce_mean(correct)
			tf.summary.scalar(scope.name+'/accuracy', accuracy)
		return accuracy
	
	def num_correct_prediction(self, logits, labels):
		correct = tf.nn.in_top_k(logits, labels, 1)
		correct = tf.cast(correct, tf.float16)
		n_correct = tf.reduce_sum(correct)
		return n_correct


class SimpleCNN(Model):
	def __init__(self, is_trainning):
		Model.__init__(self, is_trainning)

	def inference(self, x, n_classes):
		x = self.conv('conv1', x, 32, kernel_size = [3, 3], stride = [1, 1, 1, 1])
		x = self.pool('pooling1', x, kernel = [1, 3, 3, 1], stride = [1, 2, 2, 1])
		x = self.conv('conv2', x, 32, kernel_size = [3, 3], stride = [1, 1, 1, 1])
		x = self.pool('pooling2', x, kernel = [1, 3, 3, 1], stride = [1, 2, 2, 1])
		x = self.conv('conv3', x, 64, kernel_size = [3, 3], stride = [1, 1, 1, 1])
		x = self.pool('pooling3', x, kernel = [1, 3, 3, 1], stride = [1, 2, 2, 1])
		x = self.dropout("dropout1", x)
		x = self.fc_layer('fc1', x, 128)
		x = self.dropout("dropout2", x)
		x = self.fc_layer('fc2', x, 128)
		x = self.softmax_linear('output', x, n_classes)
		return x

class InceptionResnetV2(Model):
	def __init__(self, is_trainning, is_pretrain = False):
		Model.__init__(self, is_trainning)
		self.is_pretrain = is_pretrain

	def inference(self, x, n_classes):
		with slim.arg_scope(inception_resnet_v2_arg_scope()):
			x, end_points = inception_resnet_v2(x, num_classes = n_classes, is_training = False)	

		return x

	def generate_bottlenecks(self, x):
		with slim.arg_scope(inception_resnet_v2_arg_scope()):
			x, end_points = inception_resnet_v2(x, num_classes = define.N_CLASSES, is_training = False)	

		return end_points['PreLogitsFlatten']

	def inference_with_bottlenecks(self, x, n_classes):
		x = self.fc_layer('fc', x, out_nodes=1024, is_pretrain = False)		
		x = self.dropout("dropout", x)
		x = self.softmax_linear('output', x, n_classes)		
		return x

	def load(self, session, is_bottlenecks = False):
		checkpoint_file = './inception_resnet_v2_2016_08_30.ckpt'
		define.log("We will load pre-trained model from %s... " % checkpoint_file)	
		
		#Define the scopes that you want to exclude for restoration
		exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits' ]
		variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

		saver = tf.train.Saver(variables_to_restore)
		saver.restore(session, checkpoint_file)		
		define.log("model is loaded.")


class VGG16(Model):
	def __init__(self, is_trainning, is_pretrain = False):
		Model.__init__(self, is_trainning)
		self.is_pretrain = is_pretrain
		self.bottlenecks = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4']

	def load(self, session, is_bottlenecks):
		path = define.PRETRAIN_DATA_PATH
		path = os.path.join(path, "vgg19.npy")
		data_path = path
		define.log("We will load pre-trained model from %s... " % path)	

		data_dict = np.load(data_path, encoding='latin1').item()
		keys = sorted(data_dict.keys())
		for key in keys:
			if is_bottlenecks and key in self.bottlenecks:
				with tf.variable_scope(key, reuse=True):
					define.log("loading layer: %s" % key)
					for subkey, data in zip(('weights', 'biases'), data_dict[key]):
						session.run(tf.get_variable(subkey).assign(data))
			elif not is_bottlenecks and key not in self.bottlenecks:
				with tf.variable_scope(key, reuse=True):
					define.log("loading layer: %s" % key)
					for subkey, data in zip(('weights', 'biases'), data_dict[key]):
						session.run(tf.get_variable(subkey).assign(data))
		define.log("model is loaded.")

	def generate_bottlenecks(self, x):
		x = self.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv3_4', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv4_4', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.pool('pool4', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		#x = self.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		#x = self.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		#x = self.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		#x = self.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
		return x

	def inference_with_bottlenecks(self, x, n_classes):
		x = self.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = False)
		x = self.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = False)
		x = self.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = False)
		x = self.conv('conv5_4', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = False)
		x = self.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.fc_layer('fc6', x, out_nodes=4096, is_pretrain = False)
		x = self.fc_layer('fc7', x, out_nodes=4096, is_pretrain = False)
		x = self.fc_layer('fc8', x, out_nodes=1000, is_pretrain = False)		
		x = self.dropout("dropout1", x)
		x = self.fc_layer('fc9', x, out_nodes=512, is_pretrain = False)		
		x = self.dropout("dropout2", x)
		x = self.fc_layer('fc10', x, out_nodes=256, is_pretrain = False)		
		x = self.softmax_linear('output', x, n_classes)		
		return x

	def inference(self, x, n_classes):
		x = self.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain = self.is_pretrain)
		x = self.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.fc_layer('fc6', x, out_nodes=4096, is_pretrain = self.is_pretrain)
		x = self.fc_layer('fc7', x, out_nodes=4096, is_pretrain = self.is_pretrain)
		x = self.fc_layer('fc8', x, out_nodes=1000, is_pretrain = False)		
		x = self.dropout("dropout1", x)
		x = self.fc_layer('fc9', x, out_nodes=512)		
		x = self.dropout("dropout2", x)
		x = self.fc_layer('fc10', x, out_nodes=256)		
		x = self.softmax_linear('output', x, n_classes)		
		return x
	

def get_model(is_trainning, is_pretrain = False):
	model = None
	if (define.USE_MODEL == 'simple_cnn'):
		model = SimpleCNN(is_trainning = is_trainning)
	elif (define.USE_MODEL == 'vgg16'):
		model = VGG16(is_trainning = is_trainning, is_pretrain = is_pretrain)
	elif (define.USE_MODEL == 'inception_resnet_v2'):
		model = InceptionResnetV2(is_trainning = is_trainning, is_pretrain = is_pretrain)
	else:
		define.log("Unrecorgnized model = %s" % define.USE_MODEL)
	return model


def show_value():
    data_path = './vgg19.npy'

    data_dict = np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print('\n')
        print(key)
        print('weights shape: ', weights.shape)
        print('biases shape: ', biases.shape)


def main():

	checkpoint_path = './inception_resnet_v2_2016_08_30.ckpt'
	reader = tf.train.NewCheckpointReader(checkpoint_path)
	var_to_shape_map = reader.get_variable_to_shape_map()
	# Print tensor name and values
	for key in var_to_shape_map:
		print("tensor_name: ", key)
		print(reader.get_tensor(key))

	#model = get_model(True)
	
	


if __name__ == '__main__':
	main()

