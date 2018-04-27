#
# Test the model by test data or by some given image data
# 
# by ZhangChengKe
# 2017.12.28
# =======================================================

import argparse
import tensorflow as tf
import data_processing
import sys
import numpy as np
from PIL import Image
import define
import time
import math
import h5py
from model import get_model

FLAGS = None

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

def test_for_given_image(log_dir, image_file):

	im = Image.open(image_file)
	im = im.resize([define.IMAGE_W, define.IMAGE_H])
	image_array = np.array(im)
	model = get_model(False)

	with tf.Graph().as_default():
		BATCH_SIZE = 1
		N_CLASSES = 2
	
		image = tf.cast(image_array, tf.float32)
		image = tf.image.per_image_standardization(image)
		image = tf.reshape(image, [1, define.IMAGE_W, define.IMAGE_H, 3])
		logit = model.inference(image, N_CLASSES)

		logit = tf.nn.softmax(logit)

		x = tf.placeholder(tf.float32, shape=[define.IMAGE_W, define.IMAGE_H, 3])

		logs_train_dir = log_dir
		saver = tf.train.Saver()

		with tf.Session() as sess:

			if (vars(FLAGS)['silence'] != True):
				define.log("Reading checkpoints...")
			ckpt = tf.train.get_checkpoint_state(logs_train_dir)
			if ckpt and ckpt.model_checkpoint_path:
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
				saver.restore(sess, ckpt.model_checkpoint_path)
				if (vars(FLAGS)['silence'] != True):
					define.log('Loading success, global_step is %s' % global_step)
			else:
				define.log('No checkpoint file found')

			prediction = sess.run(logit, feed_dict={x: image_array})
			max_index = np.argmax(prediction)
			if max_index==0:
				if (vars(FLAGS)['silence'] != True):
					define.log('This is a cat with possibility %.6f' %prediction[:, 0])
			else:
				if (vars(FLAGS)['silence'] != True):
					define.log('This is a dog with possibility %.6f' %prediction[:, 1])	

def do_test_by_bottlenecks(test_bottlenecks, epoch = -1, num_batch = -1):
	log_dir = vars(FLAGS)['log_dir']
	is_silence = vars(FLAGS)['silence']

	if (epoch != -1):
		define.log("We will test our model after %d epoch." % epoch)
	with tf.Graph().as_default():
		num_step = int(len(test_bottlenecks) / 2)
		if num_batch != -1:
			num_step = num_batch        
		model = get_model(False)

		x = tf.placeholder(tf.float32, shape = define.BOTTLENECKS_SHAPE, name = "x")
		y = tf.placeholder(tf.int32, shape = [define.BATCH_SIZE], name = "y")

		logits = model.inference_with_bottlenecks(x, define.N_CLASSES)
		logits = tf.nn.softmax(logits)
		loss = model.losses(logits, y)
		correct = model.num_correct_prediction(logits, y)

		predictions = []
		probablities = []
		saver = tf.train.Saver(tf.global_variables())
		with tf.Session() as sess:
			define.log("Reading checkpoints...")
			ckpt = tf.train.get_checkpoint_state(log_dir)
			if (epoch == -1):
				if ckpt and ckpt.model_checkpoint_path:
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					saver.restore(sess, ckpt.model_checkpoint_path)
					if (is_silence != True):
						define.log('Loading success, global_step is %s' % global_step)
				else:
					define.log('No checkpoint file found')
			else:
				if ckpt and ckpt.all_model_checkpoint_paths:
					if epoch > len(ckpt.all_model_checkpoint_paths):
						define.log("invalid epoch = %d" % epoch)
						return None, None
					model_path = ckpt.all_model_checkpoint_paths[epoch]
					global_step = model_path.split('/')[-1].split('-')[-1]
					saver.restore(sess, model_path)
					if (is_silence != True):
						define.log('Loading success, global_step is %s' % global_step)
				else:
					define.log('No checkpoint file found')

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess = sess, coord = coord)

			try:
				num_sample = num_step*define.BATCH_SIZE
				define.log('Evaluating the model with %d images......' % num_sample)
				step = 0
				total_correct = 0
				total_loss = 0.0
				while step < num_step and not coord.should_stop():

					bottlenecks_batch = test_bottlenecks['bottlnecks_batch%d' % step]
					labels_batch = test_bottlenecks['labels_batch%d' % step]
					batch_prob, batch_correct, batch_loss = sess.run([logits, correct, loss], feed_dict = {x:bottlenecks_batch, y:labels_batch})
					predictions.extend([np.argmax(p) for p in batch_prob])
					probablities.extend([p[define.DOG] for p in batch_prob])
					total_correct += np.sum(batch_correct)
					total_loss += batch_loss
					step += 1
				define.log('Total testing samples: %d' %num_sample)
				define.log('Total correct predictions: %d' %total_correct)
				define.log('Total loss: %.3f' % (total_loss / num_step))
				define.log('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
			except Exception as e:
				coord.request_stop(e)
			finally:
				coord.request_stop()
				coord.join(threads)		

	return predictions, probablities


def do_test(images_list, labels_list, epoch = -1):
	log_dir = vars(FLAGS)['log_dir']

	if (epoch != -1):
		define.log("We will test our model after %d epoch." % epoch)
	with tf.Graph().as_default():
		if labels_list == None:
			labels_list = [0 for i in range(len(images_list))]
		model = get_model(False)
		image_batch, label_batch = data_processing.get_batches(images_list, labels_list, define.BATCH_SIZE, define.IMAGE_W, define.IMAGE_H, is_shuffle = False)
	
		logits = model.inference(image_batch, define.N_CLASSES)
		logits = tf.nn.softmax(logits)
		correct = model.num_correct_prediction(logits, label_batch)

		predictions = []
		probablities = []
		saver = tf.train.Saver(tf.global_variables())
		with tf.Session() as sess:
			define.log("Reading checkpoints...")
			ckpt = tf.train.get_checkpoint_state(log_dir)
			if (epoch == -1):
				if ckpt and ckpt.model_checkpoint_path:
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					saver.restore(sess, ckpt.model_checkpoint_path)
					if (vars(FLAGS)['silence'] != True):
						define.log('Loading success, global_step is %s' % global_step)
				else:
					define.log('No checkpoint file found')
			else:
				if ckpt and ckpt.all_model_checkpoint_paths:
					if epoch > len(ckpt.all_model_checkpoint_paths):
						define.log("invalid epoch = %d" % epoch)
						return None, None
					model_path = ckpt.all_model_checkpoint_paths[epoch]
					global_step = model_path.split('/')[-1].split('-')[-1]
					saver.restore(sess, model_path)
					if (vars(FLAGS)['silence'] != True):
						define.log('Loading success, global_step is %s' % global_step)
				else:
					define.log('No checkpoint file found')

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess = sess, coord = coord)

			try:
				define.log('Evaluating the model with %d images......' % len(images_list))
				num_step = int(math.ceil(len(images_list) / define.BATCH_SIZE))
				num_sample = num_step*define.BATCH_SIZE
				step = 0
				total_correct = 0
				while step < num_step and not coord.should_stop():
					batch_prob, batch_correct = sess.run([logits, correct])
					predictions.extend([np.argmax(p) for p in batch_prob])
					probablities.extend([p[define.DOG] for p in batch_prob])
					total_correct += np.sum(batch_correct)
					step += 1
				define.log('Total testing samples: %d' %num_sample)
				define.log('Total correct predictions: %d' %total_correct)
				define.log('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
			except Exception as e:
				coord.request_stop(e)
			finally:
				coord.request_stop()
				coord.join(threads)		

	return predictions, probablities

def main(_):

	log_dir = vars(FLAGS)['log_dir']
	test_image = vars(FLAGS)['test_image']
	test_type = vars(FLAGS)['type']
	epoch = vars(FLAGS)['epoch']

	#test_for_given_image(log_dir, test_image)
	#test_images_list = ['/Users/zhangchengke/ml/capstone/data/oxford-pet/images/Maine_Coon_1.jpg', '/Users/zhangchengke/ml/capstone/data/oxford-pet/images/Maine_Coon_2.jpg']
	#test_images_list = ['/Users/zhangchengke/ml/capstone/data/oxford-pet/images/Maine_Coon_1.jpg']
	#test_labels_list = [1, 1]
	#test_labels_list = [1]

	if (test_type == define.TYPE_ONE_IMAGE):
		test_for_given_image(log_dir, test_image)

	elif (test_type == define.TYPE_VERIFY_SET_BY_BOTTLENECKS):
		if (vars(FLAGS)['silence'] != True):
			define.log("We will evaluate model by verify data set...")
		verify_bottlenecks = h5py.File("bottlenecks_verify.hdf5", 'r')

		predictions, probalities = do_test_by_bottlenecks(verify_bottlenecks, epoch = epoch)

	elif (test_type == define.TYPE_TEST_SET_BY_BOTTLENECKS):
		if (vars(FLAGS)['silence'] != True):
			define.log("We will evaluate our model by test data set...")
	
		test_images_list = data_processing.get_test_data_from_kaggle_dataset(define.DATA_DIR)
		test_bottlenecks = h5py.File("bottlenecks_test.hdf5", 'r')
		predictions, probalities = do_test_by_bottlenecks(test_bottlenecks, epoch = epoch)
		time_stamp = time.strftime("%m%d%H%M%S", time.localtime())
		f = open("dogs_vs_cats_submission_%s.csv" % time_stamp, "w")
		f.write("id,label\n")
		for image, p in zip(test_images_list, probalities):
			p = p.clip(min=0.005, max=0.995)
			f.write("%s,%.3f\n" % (image.split('/')[-1].split('.')[0], p))
		f.close()

	elif (test_type == define.TYPE_TEST_SET):
		if (vars(FLAGS)['silence'] != True):
			define.log("We will evaluate our model by test data set...")
		test_images_list = data_processing.get_test_data_from_kaggle_dataset(define.DATA_DIR)
	
		#test_images_list = test_images_list[:8]
		predictions, probalities = do_test(test_images_list, None, epoch = epoch)
		time_stamp = time.strftime("%m%d%H%M%S", time.localtime())
		f = open("dogs_vs_cats_submission_%s.csv" % time_stamp, "w")
		f.write("id,label\n")
		for image, p in zip(test_images_list, probalities):
			f.write("%s,%.3f\n" % (image.split('/')[-1].split('.')[0], p))
		f.close()

	elif (test_type == define.TYPE_EPOCH_VERIFY_SET):
		if (vars(FLAGS)['silence'] != True):
			define.log("We will evaluate model of each epoch by verify data set...")
		test_data_list = log_dir + '/test_list.csv'
		test_images_list, test_labels_list = data_processing.load_list(test_data_list)

		#test_images_list = test_images_list[:8]
		#test_labels_list = test_labels_list[:8]
		for epoch in range(define.N_EPOCH):
			predictions, probalities = do_test(test_images_list, test_labels_list, epoch)
			image_cnt = len(test_images_list)
			ok_cnt = get_ok_cnt(test_labels_list, predictions)
			log_loss = get_log_loss(test_labels_list, probalities)

			define.log("****** EPOCH %d: AVERAGE ACCURCY = %.6f, OK COUNT = %d, LOG LOSS = %.6f  *******" % (epoch, ok_cnt / image_cnt, ok_cnt, log_loss))

	elif (test_type == define.TYPE_VERIFY_SET):
		if (vars(FLAGS)['silence'] != True):
			define.log("We will evaluate model by verify data set...")
		test_data_list = log_dir + '/test_list.csv'
		test_images_list, test_labels_list = data_processing.load_list(test_data_list)

		#test_images_list = test_images_list[:16]
		#test_labels_list = test_labels_list[:16]
		predictions, probalities = do_test(test_images_list, test_labels_list, epoch = epoch)
		image_cnt = len(test_images_list)
		ok_cnt = get_ok_cnt(test_labels_list, predictions)
		log_loss = get_log_loss(test_labels_list, probalities)

		define.log("****** AVERAGE ACCURCY = %.6f, OK COUNT = %d, LOG LOSS = %.6f  *******" % (ok_cnt / image_cnt, ok_cnt, log_loss))
	else:
		define.log("Unrecognized type = %s" % test_type)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir', type=str, default=define.LOG_DIR,
                      help='Directory for storing logs data')
	parser.add_argument('--test_image', type=str, default="",
                      help='A given image for test')
	parser.add_argument('--type', type=str, default=define.TYPE_VERIFY_SET,
                      help='Test type, should be the values as like (verify, test, image, epoch)')
	parser.add_argument('--silence', action='store_true',
						help='Whether output debug log or not.')
	parser.add_argument('--epoch', type=int, default=-1,
                      help='A given epoch for test')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
