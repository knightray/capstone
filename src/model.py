#
# Build a CNN model based on cifar-10 example
# 
# by ZhangChengKe
# 2017.12.28
# =======================================================

import tensorflow as tf


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
		

class SimpleCNN(Model):
	def __init__(self):
		pass

	def inference(self, x, batch_size, n_classes):
		x = self.conv('conv1', x, 16, kernel_size = [3, 3], stride = [1, 1, 1, 1])
		x = self.pool('pooling1', x, kernel = [1, 3, 3, 1], stride = [1, 2, 2, 1])
		x = self.conv('conv2', x, 16, kernel_size = [3, 3], stride = [1, 1, 1, 1])
		x = self.pool('pooling2', x, kernel = [1, 3, 3, 1], stride = [1, 2, 2, 1])
		x = self.fc_layer('fc1', x, 128)
		x = self.fc_layer('fc2', x, 128)
		x = self.softmax_linear('output', x, n_classes)
		return x


def simple_cnn(images, batch_size, n_classes):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]

    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape = [3,3,3, 16],
                                  dtype = tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)

    #pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm1')

    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,16,16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')


    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')


    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        print("reshape = %s, dim = %d" % (reshape, dim))
        print("pool2.shape=%s" % pool2.shape)
        weights = tf.get_variable('weights',
                                  shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    #local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')


    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear


scnn = SimpleCNN()
def inference(images, batch_size, n_classes):
	return scnn.inference(images, batch_size, n_classes)
	#return simple_cnn(images, batch_size, n_classes)

#%%
def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

#%%
def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy

#%%

def main():
	pass


if __name__ == '__main__':
	main()

