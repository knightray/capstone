#
# Build a CNN model based on cifar-10 example
# 
# by ZhangChengKe
# 2017.12.28
# =======================================================

import tensorflow as tf
import define


class Model(object):
	def __init__(self):
		pass

	def conv(self, layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True):
		in_channels = x.get_shape()[-1]
		with tf.variable_scope(layer_name):
			w = tf.get_variable(name='weights',
					trainable=is_pretrain,
					shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
					initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable(name='biases',
					trainable=is_pretrain,
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

	def batch_norm(self, x):
		epsilon = 1e-3
		batch_mean, batch_var = tf.nn.moments(x, [0])
		x = tf.nn.batch_normalization(x,
				mean=batch_mean,
				variance=batch_var,
				offset=None,
				scale=None,
				variance_epsilon=epsilon)
		return x

	def fc_layer(self, layer_name, x, out_nodes):
		shape = x.get_shape()
		if len(shape) == 4:
			size = shape[1].value * shape[2].value * shape[3].value
		else:
			size = shape[-1].value

		with tf.variable_scope(layer_name):
			w = tf.get_variable('weights',
					shape=[size, out_nodes],
					initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable('biases',
					shape=[out_nodes],
					initializer=tf.constant_initializer(0.0))
			flat_x = tf.reshape(x, [-1, size]) # flatten into 1D

			x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
			x = tf.nn.relu(x)
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
	def __init__(self):
		pass

	def inference(self, x, n_classes):
		x = self.conv('conv1', x, 16, kernel_size = [3, 3], stride = [1, 1, 1, 1])
		x = self.pool('pooling1', x, kernel = [1, 3, 3, 1], stride = [1, 2, 2, 1])
		x = self.conv('conv2', x, 16, kernel_size = [3, 3], stride = [1, 1, 1, 1])
		x = self.pool('pooling2', x, kernel = [1, 3, 3, 1], stride = [1, 2, 2, 1])
		x = self.fc_layer('fc1', x, 128)
		x = self.fc_layer('fc2', x, 128)
		x = self.softmax_linear('output', x, n_classes)
		return x

class VGG16(Model):
	def __init__(self):
		pass

	def inference(self, x, n_classes):
		x = self.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
		x = self.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

		x = self.fc_layer('fc6', x, out_nodes=4096)
		x = self.batch_norm(x)
		x = self.fc_layer('fc7', x, out_nodes=4096)
		x = self.batch_norm(x)
		x = self.fc_layer('fc8', x, out_nodes=n_classes)		
		return x
	

def get_model():
	model = None
	if (define.USE_MODE == 'simple_cnn'):
		model = SimpleCNN()
	else:
		model = VGG16()
	return model


def main():
	pass


if __name__ == '__main__':
	main()

