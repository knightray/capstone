#
# Test the model by test data or by some given image data
# 
# by ZhangChengKe
# 2017.12.28
# =======================================================

import argparse
import tensorflow as tf
import data_processing
import model
import sys
import numpy as np
from PIL import Image
import define

FLAGS = None

def load_model(log_dir, sess):
	saver = tf.train.import_meta_graph('%s/model.ckpt-10000.meta' % log_dir)
	saver.restore(sess, '%s/model.ckpt-10000' % log_dir)

def test_given_image2(log_dir, image_file):

	with tf.Graph().as_default():
		BATCH_SIZE = 1
		N_CLASSES = 2
	
		image = tf.image.decode_jpeg(tf.read_file(image_file), try_recover_truncated = True, acceptable_fraction = 0.5, channels = 3)
		image = tf.image.resize_image_with_crop_or_pad(image, define.IMAGE_W, define.IMAGE_H)
		image = tf.image.per_image_standardization(image)
		image = tf.reshape(image, [1, define.IMAGE_W, define.IMAGE_H, 3])
		logit = model.inference(image, BATCH_SIZE, N_CLASSES)

		logit = tf.nn.softmax(logit)

		logs_train_dir = log_dir
		saver = tf.train.Saver()

		with tf.Session() as sess:

			print("Reading checkpoints...")
			ckpt = tf.train.get_checkpoint_state(logs_train_dir)
			if ckpt and ckpt.model_checkpoint_path:
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
				saver.restore(sess, ckpt.model_checkpoint_path)
				print('Loading success, global_step is %s' % global_step)
			else:
				print('No checkpoint file found')

			prediction = sess.run(logit)
			max_index = np.argmax(prediction)
			if max_index==0:
				print('This is a cat with possibility %.6f' %prediction[:, 0])
			else:
				print('This is a dog with possibility %.6f' %prediction[:, 1])	

def test_given_image(log_dir, image_file):

	print("log_dir=%s, image_file=%s" % (log_dir, image_file))
	im = Image.open(image_file)
	im = im.resize([define.IMAGE_W, define.IMAGE_H])
	image_array = np.array(im)

	with tf.Graph().as_default():
		BATCH_SIZE = 1
		N_CLASSES = 2
	
		image = tf.cast(image_array, tf.float32)
		image = tf.image.per_image_standardization(image)
		image = tf.reshape(image, [1, define.IMAGE_W, define.IMAGE_H, 3])
		logit = model.inference(image, BATCH_SIZE, N_CLASSES)

		logit = tf.nn.softmax(logit)

		x = tf.placeholder(tf.float32, shape=[define.IMAGE_W, define.IMAGE_H, 3])

		logs_train_dir = log_dir
		saver = tf.train.Saver()

		with tf.Session() as sess:

			print("Reading checkpoints...")
			ckpt = tf.train.get_checkpoint_state(logs_train_dir)
			if ckpt and ckpt.model_checkpoint_path:
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
				saver.restore(sess, ckpt.model_checkpoint_path)
				print('Loading success, global_step is %s' % global_step)
			else:
				print('No checkpoint file found')

			prediction = sess.run(logit, feed_dict={x: image_array})
			max_index = np.argmax(prediction)
			if max_index==0:
				print('This is a cat with possibility %.6f' %prediction[:, 0])
			else:
				print('This is a dog with possibility %.6f' %prediction[:, 1])	

def main(_):

	log_dir = vars(FLAGS)['log_dir']
	test_image = vars(FLAGS)['test_image']

	test_given_image2(log_dir, test_image)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir', type=str, default=define.LOG_DIR,
                      help='Directory for storing logs data')
	parser.add_argument('--test_image', type=str, default="",
                      help='A given image for test')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
