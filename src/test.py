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

def get_test_image(image_file):
	image = tf.image.decode_jpeg(tf.read_file(image_file), try_recover_truncated = True, acceptable_fraction = 0.5, channels = 3)
	image = tf.image.resize_image_with_crop_or_pad(image, define.IMAGE_W, define.IMAGE_H)
	images = tf.image.per_image_standardization(images)
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
	#print("images_array = %s" % images_array.shape)
	return images_array

def is_dog_or_cat(label):
	return 'CAT' if label == define.CAT else 'DOG'

def get_log_loss(labels, predications):
	log_loss = tf.losses.log_loss(labels, predications)
	with tf.Session() as sess:
		log_loss_val = sess.run(log_loss)
	return log_loss_val

def get_ok_cnt(labels, predications):
	ok_cnt = 0
	for p, l in zip(predications, labels):
		if p == l:
			ok_cnt += 1
	
	return ok_cnt

def test_for_verify_data(log_dir, images_list, labels_list):

	with tf.Graph().as_default():
		BATCH_SIZE = len(images_list)
		N_CLASSES = 2

		image_batch = read_images(images_list)
		logit = model.inference(image_batch, BATCH_SIZE, N_CLASSES)
		logit = tf.nn.softmax(logit)

		logs_train_dir = log_dir
		saver = tf.train.Saver()

		with tf.Session() as sess:

			#print("Reading checkpoints...")
			ckpt = tf.train.get_checkpoint_state(logs_train_dir)
			if ckpt and ckpt.model_checkpoint_path:
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
				saver.restore(sess, ckpt.model_checkpoint_path)
				#print('Loading success, global_step is %s' % global_step)
			else:
				print('No checkpoint file found')

			predictions = sess.run(logit)
			if (labels_list != None):
				for p, label, image in zip(predictions, labels_list, images_list):
					max_index = np.argmax(p)
					is_ok = "OK" if max_index == label else "NG"
					print("%-40s [%s] - [%s] - with possibility %s" % (image.split('/')[-1], is_dog_or_cat(label), is_ok, p[define.DOG]))

	pred = [np.argmax(p) for p in predictions]	
	prob = [p[define.DOG] for p in predictions]
	return pred, prob

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

def do_test(images_list, labels_list):
	log_dir = vars(FLAGS)['log_dir']

	print(images_list)
	batch_size = 8
	ok_cnt = 0
	image_cnt = len(images_list)
	loop_cnt = int(image_cnt / batch_size)
	if image_cnt % batch_size != 0:
		loop_cnt += 1

	predictions = []
	probablities = []
	for i in range(loop_cnt):
		print("*** Testing batch %d, image from %d to %d... ***" % (i, i * batch_size, min((i + 1) * batch_size, image_cnt)))
		image_batch = images_list[i * batch_size : min((i + 1) * batch_size, image_cnt)]
		if (labels_list != None):
			label_batch = labels_list[i * batch_size : min((i + 1) * batch_size, image_cnt)]
		else:
		 	label_batch = None
		pred_batch, prob_batch =  test_for_verify_data(log_dir, image_batch, label_batch)
		predictions.extend(pred_batch)
		probablities.extend(prob_batch)

	return predictions, probablities

def main(_):

	log_dir = vars(FLAGS)['log_dir']
	test_image = vars(FLAGS)['test_image']
	test_set = vars(FLAGS)['test_set']

	#test_for_given_image(log_dir, test_image)
	#test_images_list = ['/Users/zhangchengke/ml/capstone/data/oxford-pet/images/Maine_Coon_1.jpg', '/Users/zhangchengke/ml/capstone/data/oxford-pet/images/Maine_Coon_2.jpg']
	#test_images_list = ['/Users/zhangchengke/ml/capstone/data/oxford-pet/images/Maine_Coon_1.jpg']
	#test_labels_list = [1, 1]
	#test_labels_list = [1]

	if (test_image != ""):
		test_for_given_image(log_dir, test_image)

	elif (test_set != ""):
		print("We will evaluate our model by test data set...")
		test_images_list = data_processing.get_test_data_from_kaggle_dataset(test_set)
	
		#test_images_list = test_images_list[:8]
		predictions, probalities = do_test(test_images_list, None)
		f = open("dogs_vs_cats_submission.csv", "w")
		f.write("id,label\n")
		for image, p in zip(test_images_list, probalities):
			f.write("%s,%.1f\n" % (image.split('/')[-1].split('.')[0], p))
		f.close()

	else:
		test_data_list = log_dir + '/test_list.csv'
		test_images_list, test_labels_list = data_processing.load_list(test_data_list)

		#test_images_list = test_images_list[:8]
		#test_labels_list = test_labels_list[:8]
		predictions, probalities = do_test(test_images_list, test_labels_list)
		ok_cnt = get_ok_cnt(test_labels_list, predictions)
		log_loss = get_log_loss(test_labels_list, probablities)

		print("****** AVERAGE ACCURCY = %.6f, OK COUNT = %d, LOG LOSS = %.6f  *******" % (ok_cnt / image_cnt, ok_cnt, log_loss))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir', type=str, default=define.LOG_DIR,
                      help='Directory for storing logs data')
	parser.add_argument('--test_image', type=str, default="",
                      help='A given image for test')
	parser.add_argument('--test_set', type=str, default="",
                      help='Images path for test')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
