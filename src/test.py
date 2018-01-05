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

def get_test_image(image_file):
	image = tf.image.decode_jpeg(tf.read_file(image_file), try_recover_truncated = True, acceptable_fraction = 0.5, channels = 3)
	image = tf.image.resize_image_with_crop_or_pad(image, define.IMAGE_W, define.IMAGE_H)
	#images = tf.image.per_image_standardization(images)
	image = tf.cast(image, tf.float32)
	image = tf.reshape(image, [1, define.IMAGE_W, define.IMAGE_H, 3])
	return image

def read_images(images_list):
    
	images_array = None
	for image_file in images_list:
		im = Image.open(image_file)
		im = im.resize([define.IMAGE_W, define.IMAGE_H])
		image_data = np.array(im)
		image = tf.cast(image_data, tf.float32)
		image = tf.image.per_image_standardization(image)   

		if (images_array == None):
			images_array = image
		else:
			images_array = tf.concat([images_array, image], 0)

	images_array = tf.reshape(images_array, [-1, define.IMAGE_W, define.IMAGE_H, 3])        
	print("images_array = %s" % images_array.shape)
	return images_array

def is_dog_or_cat(label):
	return 'cat' if label == 0 else 'dog'

def get_accurcy(images, labels, predictions):

	accurcy = 0.0
	for image, label, p in zip(images, labels, predictions):
		accurcy += p[label]
		max_index = np.argmax(p)
		if label == max_index:
			print("%s [OK][%s] - with possibility %.6f" % (image, is_dog_or_cat(label), p[max_index]))
		else:
			print("%s [NG][%s] - with possibility %.6f" % (image, is_dog_or_cat(label), p[max_index]))

	print("****** AVERAGE ACCURCY = %.6f *******" % (accurcy / len(images)))


def test_for_test_data(log_dir, images_list, labels_list):

	with tf.Graph().as_default():
		BATCH_SIZE = len(images_list)
		N_CLASSES = 2

		image_batch = read_images(images_list)
		logit = model.inference(image_batch, BATCH_SIZE, N_CLASSES)
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

			predictions = sess.run(logit)
			get_accurcy(images_list, labels_list, predictions)

def test_for_given_image(log_dir, image_file):

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

	#test_for_given_image(log_dir, test_image)
	#test_images_list = ['/Users/zhangchengke/ml/capstone/data/oxford-pet/images/Maine_Coon_1.jpg', '/Users/zhangchengke/ml/capstone/data/oxford-pet/images/Maine_Coon_2.jpg']
	#test_images_list = ['/Users/zhangchengke/ml/capstone/data/oxford-pet/images/Maine_Coon_1.jpg']
	#test_labels_list = [1, 1]
	#test_labels_list = [1]

	train_images_list, train_labels_list, test_images_list, test_labels_list = data_processing.get_files_from_oxford_pet_dataset(define.DATA_DIR)
	test_for_test_data(log_dir, test_images_list, test_labels_list)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir', type=str, default=define.LOG_DIR,
                      help='Directory for storing logs data')
	parser.add_argument('--test_image', type=str, default="",
                      help='A given image for test')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
