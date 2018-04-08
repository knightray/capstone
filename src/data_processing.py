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
import csv
from PIL import Image

FLAGS = None

def load_list(file_name):
	images = []
	labels = []

	f = open(file_name, 'r')
	lines = f.readlines()
	for line in lines:
		tokens = line.split(',')
		images.append(tokens[0])
		labels.append(int(tokens[1]))
	f.close()
	return images, labels

def save_list(images, labels, file_name):
	f = open(file_name, 'w')
	for image, label in zip(images, labels):
		f.write("%s,%d\n" % (image, label))
	f.close()

def get_test_data_from_kaggle_dataset(data_dir):
	images_dir = data_dir + 'test/'

	images_list = []
	files = os.listdir(images_dir)
	for f in files:
		images_list.append(images_dir + f)

	images_list = sorted(images_list, key = lambda d : int(d.split('/')[-1].split('.')[0]))
	return images_list

def get_train_data_from_kaggle_dataset(data_dir):
	images_dir = data_dir + 'train/'

	images_list = []
	labels_list = []
	all_files = []

	files = os.listdir(images_dir)
	for f in files:
		label_str = f.split(".")[0]
		if label_str == 'cat':
			labels_list.append(define.CAT)
		elif label_str == 'dog':
			labels_list.append(define.DOG)
		else:
			print("Error: unrecognized type = %s" % label_str)
		images_list.append(images_dir + f)

	tmp = np.array([images_list, labels_list])
	tmp = tmp.transpose()
	np.random.shuffle(tmp)

	images_list = list(tmp[:, 0])
	labels_list = list(tmp[:, 1])
	labels_list = [int(l) for l in labels_list]

	total_cnt = len(images_list)
	train_cnt = int(total_cnt * define.TRAINING_IMAGE_PERCENT)
	return images_list[:train_cnt], labels_list[:train_cnt], images_list[train_cnt:], labels_list[train_cnt:]

def get_files_from_oxford_pet_dataset(data_dir):
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
	train_cnt = int(len(images_list) * define.TRAINING_IMAGE_PERCENT)

	return images_list[:train_cnt], labels_list[:train_cnt], images_list[train_cnt + 1:], labels_list[train_cnt + 1:]


def get_batches(images_list, labels_list, batch_size, image_width, image_height, is_shuffle = True):

	images = tf.cast(images_list, tf.string)
	labels = tf.cast(labels_list, tf.int32)

	input_queue = tf.train.slice_input_producer([images, labels], shuffle = is_shuffle)
	labels = input_queue[1]
	images = tf.image.decode_jpeg(tf.read_file(input_queue[0]), try_recover_truncated = True, acceptable_fraction = 0.5, channels = 3)
	#print("images.shape=%s" % tf.shape(images)[0])

	# Resize the image twice in order to avoid the image distortion
	max_size = tf.maximum(tf.shape(images)[0], tf.shape(images)[1])
	images = tf.image.resize_image_with_crop_or_pad(images, max_size, max_size)
	images = tf.image.resize_image_with_crop_or_pad(images, image_width, image_height)
	#images = tf.image.resize_images(images, [image_width, image_height], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	#images = tf.image.resize_images(images, [image_width, image_height])
	#images = tf.image.random_brightness(images, max_delta=0.5)
	#images = tf.image.random_contrast(images, lower = 0.1, upper = 0.8)
	#images = tf.image.random_flip_left_right(images)
	images = tf.image.per_image_standardization(images)

	image_batch, label_batch = tf.train.batch([images, labels], batch_size = batch_size, num_threads = 1, capacity = len(images_list))
	label_batch = tf.reshape(label_batch, [batch_size])
	image_batch = tf.cast(image_batch, tf.float32)

	return image_batch, label_batch
	
def get_image_info(data_dir):

	images_dir = data_dir + 'train/'
	images = os.listdir(images_dir)

	all_w = []
	all_h = []
	for image in images:
		im = Image.open(images_dir + image)
		#print("%s -> %s" % (image, im.size))
		all_w.append(im.size[0])
		all_h.append(im.size[1])
		im.close()

	with open("imageinfo.csv", 'w', newline='') as f:
		writer = csv.writer(f)
		for image, w, h in zip(images, all_w, all_h):
			w_h = "%.2f" % (w/h)
			writer.writerow([w, h, w_h])

	print("max_w = %d, max_h = %d" % (max(all_w), max(all_h)))
	

def main(_):

	import matplotlib.pyplot as plt

	data_dir = vars(FLAGS)['data_dir']
	output_dir = data_dir + "output/"
	batch_size = 8
	batch_num = 2
	image_w = 500
	image_h = 500

	#test_images_list = get_test_data_from_kaggle_dataset(data_dir)
	#test_images_list = test_images_list[:16]
	#test_labels_list = [0 for i in range(16)]
	#print(test_images_list)
	train_images_list, train_labels_list, test_images_list, test_labels_list = get_train_data_from_kaggle_dataset(data_dir)	
	#train_images_list = train_images_list[:10]
	#train_labels_list = train_labels_list[:10]
	print("We got %d images for training, %d images for test." % (len(train_images_list), len(test_images_list)))

	get_image_info(data_dir)

	image_batch, label_batch = get_batches(train_images_list, train_labels_list, batch_size, image_w, image_h)
	#image_batch, label_batch = get_batches(test_images_list, test_labels_list, batch_size, image_w, image_h)
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
					filepath = output_dir + "batch%d_%d.png" % (i, j)
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

