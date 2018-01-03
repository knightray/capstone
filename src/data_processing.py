#
# Read images data from files and resize all images to 
# the same size in order to feed the model as the data
# input.
# 
# by ZhangChengKe
# 2017.12.28
# =======================================================

import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import define

FLAGS = None
TRAINING_IMAGE_PERCENT = 0.8

def get_files_from_oxford_pet_dataset(data_dir):
	'''
	Args:
		data_dir: directory of images data
	Returns:
		list of images and labels
	'''

	images_dir = 'images/'
	labels_file = data_dir + 'annotations/list.txt'

	images_list = []
	labels_list = []
	all_files = []

	with open(labels_file) as lf:
		lines = lf.readlines()
		for line in lines:

			if line[0] == '#':
				continue

			tokens = line.split()
			if len(tokens) != 4:
				print("Invalid line: %s" % line)

			pet_name = tokens[0]
			class_id = int(tokens[1])
			species = int(tokens[2])
			breed_id = int(tokens[3])

			image_file = data_dir + images_dir + pet_name + ".jpg"
			if not os.path.exists(image_file):
				print("We can not find file %s" % image_file)

			images_list.append(image_file)
			labels_list.append(species)

			lf.close()

	# Shuffle the images and labels list, and divide the test and train set
	tmp = np.array([images_list, labels_list])
	tmp = tmp.transpose()
	np.random.shuffle(tmp)

	images_list = list(tmp[:, 0])
	labels_list = list(tmp[:, 1])

	# As the labels in the source data presents the classifcation like this 1: cat, 2: dog,
	# we simply substract 1 from the original value so we can get the label as 0:cat 1:dog	
	labels_list = [int(l) - 1 for l in labels_list]
	train_cnt = int(len(images_list) * TRAINING_IMAGE_PERCENT)

	return images_list[:train_cnt], labels_list[:train_cnt], images_list[train_cnt + 1:], labels_list[train_cnt + 1:]


def get_batches(images_list, labels_list, batch_size, image_width, image_height):

	images = tf.cast(images_list, tf.string)
	labels = tf.cast(labels_list, tf.int32)

	input_queue = tf.train.slice_input_producer([images, labels])
	labels = input_queue[1]
	print("decode jpeg %s" % input_queue[0])
	images = tf.image.decode_jpeg(tf.read_file(input_queue[0]), try_recover_truncated = True, acceptable_fraction = 0.5, channels = 3)

	images = tf.image.resize_image_with_crop_or_pad(images, image_width, image_height)
	#images = tf.image.per_image_standardization(images)

	image_batch, label_batch = tf.train.batch([images, labels], batch_size = batch_size, num_threads = 64)
	label_batch = tf.reshape(label_batch, [batch_size])
	image_batch = tf.cast(image_batch, tf.float32)

	return image_batch, label_batch
	

def main(_):

	import matplotlib.pyplot as plt

	data_dir = vars(FLAGS)['data_dir']
	output_dir = data_dir + "output/"
	batch_size = 10
	batch_num = 1
	image_w = 300
	image_h = 300

	train_images_list, train_labels_list, test_images_list, test_labels_list = get_files_from_oxford_pet_dataset(data_dir)	
	print("We got %d images for training, %d images for test." % (len(train_images_list), len(test_images_list)))

	image_batch, label_batch = get_batches(train_images_list, train_labels_list, batch_size, image_w, image_h)
	print ("We got image_batch=%s, label_batch=%s" % (image_batch.shape, label_batch.shape))

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	with tf.Session() as sess:
		i = 0
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord = coord)

		try:
			while not coord.should_stop() and i < batch_num:
				image, label = sess.run([image_batch, label_batch])
				for j in range(batch_size):
					print("batch %d, %d, label:%d" % (i, j, label[j]))
					plt.imshow(image[j,:,:,:])
					filepath = output_dir + "batch%d_%d_label%d.png" % (i, j, label[j])
					plt.savefig(filepath)
					#plt.show()
				i += 1
		except tf.errors.OutOfRangeError:
		   print("done!")
		finally:
		   coord.request_stop()

		coord.join(threads)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default=define.DATA_DIR,
                      help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

