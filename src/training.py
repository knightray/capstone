#
# Trainning the model by given image data
# 
# by ZhangChengKe
# 2017.12.28
# =======================================================

import argparse
import tensorflow as tf
import data_processing
import sys
import os
import h5py
import numpy as np
import define
from model import get_model 

FLAGS = None

def trainning_and_verify(train_images, train_labels, verify_images, verify_labels):

	model = get_model(True, define.USE_PRETRAIN)
	train_image_batch, train_label_batch = data_processing.get_batches(train_images, train_labels, define.BATCH_SIZE, define.IMAGE_W, define.IMAGE_H)
	verify_image_batch, verify_label_batch = data_processing.get_batches(verify_images, verify_labels, define.BATCH_SIZE, define.IMAGE_W, define.IMAGE_H)
	
	#print("image_batch=%s, label_batch=%s" % (image_batch.shape, label_batch))
	x = tf.placeholder(tf.float32, shape = [define.BATCH_SIZE, define.IMAGE_W, define.IMAGE_H, 3], name = "x")
	y = tf.placeholder(tf.int32, shape = [define.BATCH_SIZE], name = "y")

	train_logits = model.inference(x, define.N_CLASSES)
	train_loss = model.losses(train_logits, y)
	train_op = model.trainning(train_loss, define.LEARNING_RATE)
	train_acc_op = model.evaluation(train_logits, y)

	model.print_all_variables()

	logs_dir = vars(FLAGS)['log_dir']
	if (vars(FLAGS)['max_step'] != 0):
		max_step = vars(FLAGS)['max_step']
	else:
		max_step = int(len(train_images) / define.BATCH_SIZE)
	
	define.log("Do trainning for %d step in one epoch." % max_step)
	if not os.path.exists(logs_dir):
		os.mkdir(logs_dir)

	summary_op = tf.summary.merge_all()
	sess = tf.Session()
	train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
	saver = tf.train.Saver(max_to_keep=define.N_EPOCH)

	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	if (define.USE_PRETRAIN):
		model.load(sess)

	try:
		for epoch in range(define.N_EPOCH):
			for step in range(max_step):
				if coord.should_stop():
					break
				tra_images, tra_labels = sess.run([train_image_batch, train_label_batch])
				_, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc_op], feed_dict = {x:tra_images, y:tra_labels})

				if step % 100 == 0:
					define.log(' Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
					summary_str = sess.run(summary_op, feed_dict = {x:tra_images, y:tra_labels})
					train_writer.add_summary(summary_str, step)

			define.log(' END Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
			checkpoint_path = os.path.join(logs_dir, 'model_%s.ckpt' % define.USE_MODEL)
			saver.save(sess, checkpoint_path, global_step=(epoch + 1)*max_step)

			define.log(" Verify at verify set...")
			verify_images, verify_labels = sess.run([verify_image_batch, verify_label_batch])
			_, verify_loss, verify_acc = sess.run([train_op, train_loss, train_acc_op], feed_dict={x:verify_images, y:verify_labels})
			define.log(' verify loss = %.2f, verify accuracy = %.2f%%' %(verify_loss, verify_acc*100.0))

			define.log("**** EPOCH %d FINISHED ****" % (epoch + 1)) 

	except tf.errors.OutOfRangeError:
		define.log('Done training -- epoch limit reached')
	finally:
		coord.request_stop()

	coord.join(threads)
	sess.close()

def training(images, labels):

	model = get_model(True, define.USE_PRETRAIN)
	image_batch, label_batch = data_processing.get_batches(images, labels, define.BATCH_SIZE, define.IMAGE_W, define.IMAGE_H)
	
	#print("image_batch=%s, label_batch=%s" % (image_batch.shape, label_batch))
	train_logits = model.inference(image_batch, define.N_CLASSES)
	train_loss = model.losses(train_logits, label_batch)
	train_op = model.trainning(train_loss, define.LEARNING_RATE)
	train_acc_op = model.evaluation(train_logits, label_batch)

	logs_dir = vars(FLAGS)['log_dir']
	if (vars(FLAGS)['max_step'] != 0):
		max_step = vars(FLAGS)['max_step']
	else:
		max_step = int(len(images) / define.BATCH_SIZE)
	
	define.log("Do trainning for %d step in one epoch." % max_step)
	if not os.path.exists(logs_dir):
		os.mkdir(logs_dir)

	summary_op = tf.summary.merge_all()
	sess = tf.Session()
	train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
	saver = tf.train.Saver(max_to_keep=define.N_EPOCH)

	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	try:
		for epoch in range(define.N_EPOCH):
			for step in range(max_step):
				if coord.should_stop():
					break
				_, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc_op])

				if step % 100 == 0:
					define.log(' Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
					summary_str = sess.run(summary_op)
					train_writer.add_summary(summary_str, step)

			define.log(' END Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
			define.log("**** EPOCH %d FINISHED ****" % (epoch + 1)) 
			checkpoint_path = os.path.join(logs_dir, 'model_%s.ckpt' % define.USE_MODEL)
			saver.save(sess, checkpoint_path, global_step=(epoch + 1)*max_step)

	except tf.errors.OutOfRangeError:
		define.log('Done training -- epoch limit reached')
	finally:
		coord.request_stop()

	coord.join(threads)
	sess.close()

def train_by_bottlenecks(train_bottlenecks, verify_bottlenecks):

	model = get_model(True, False)
	x = tf.placeholder(tf.float32, shape = [define.BATCH_SIZE, 7, 7, 512], name = "x")
	y = tf.placeholder(tf.int32, shape = [define.BATCH_SIZE], name = "y")

	train_logits = model.inference_with_bottlenecks(x, define.N_CLASSES)
	train_loss = model.losses(train_logits, y)
	train_op = model.trainning(train_loss, define.LEARNING_RATE)
	train_acc_op = model.evaluation(train_logits, y)

	model.print_all_variables()

	logs_dir = vars(FLAGS)['log_dir']
	if (vars(FLAGS)['max_step'] != 0):
		max_step = vars(FLAGS)['max_step']
	else:
		max_step = int(len(train_bottlenecks) / 2)
	
	define.log("Do trainning for %d step in one epoch." % max_step)
	if not os.path.exists(logs_dir):
		os.mkdir(logs_dir)

	summary_op = tf.summary.merge_all()
	sess = tf.Session()
	train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
	saver = tf.train.Saver(max_to_keep=define.N_EPOCH)

	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	#if (define.USE_PRETRAIN):
	#	model.load(sess)

	try:
		for epoch in range(define.N_EPOCH):
			for step in range(max_step):
				if coord.should_stop():
					break

				tra_bottlenecks_batch = train_bottlenecks['bottlnecks_batch%d' % step]
				tra_labels_batch = train_bottlenecks['labels_batch%d' % step]
				_, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc_op], feed_dict = {x:tra_bottlenecks_batch, y:tra_labels_batch})

				if step % 100 == 0:
					define.log(' Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
					summary_str = sess.run(summary_op, feed_dict = {x:tra_bottlenecks_batch, y:tra_labels_batch})
					train_writer.add_summary(summary_str, step)

			define.log(' END Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
			checkpoint_path = os.path.join(logs_dir, 'model_%s.ckpt' % define.USE_MODEL)
			saver.save(sess, checkpoint_path, global_step=(epoch + 1)*max_step)

			define.log(" Verify at verify set...")
			verify_bottlenecks_batch = verify_bottlenecks['bottlnecks_batch0']
			verify_labels_batch = verify_bottlenecks['labels_batch0']
			_, verify_loss, verify_acc = sess.run([train_op, train_loss, train_acc_op], feed_dict={x:verify_bottlenecks_batch, y:verify_labels_batch})
			define.log(' verify loss = %.2f, verify accuracy = %.2f%%' %(verify_loss, verify_acc*100.0))

			define.log("**** EPOCH %d FINISHED ****" % (epoch + 1)) 

	except tf.errors.OutOfRangeError:
		define.log('Done training -- epoch limit reached')
	finally:
		coord.request_stop()

	coord.join(threads)
	sess.close()

def generate_bottlenecks(images, labels, typestr):
	with tf.Graph().as_default():
		model = get_model(True, define.USE_PRETRAIN)
		image_batch, label_batch = data_processing.get_batches(images, labels, define.BATCH_SIZE, define.IMAGE_W, define.IMAGE_H)
	
		bottlenecks = model.generate_bottlenecks(image_batch)
		max_step = int(len(images) / define.BATCH_SIZE)	
		#max_step = 10
		sess = tf.Session()

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		if (define.USE_PRETRAIN):
			model.load(sess)

		f = h5py.File("bottlenecks_%s.hdf5" % typestr, "w")
		try:
			for step in range(max_step):
				if coord.should_stop():
					break

				images_vals, labels_vals = sess.run([image_batch, label_batch])
				#print(labels_vals)
				bottlenecks_vals = sess.run(bottlenecks)
				bdata = f.create_dataset('bottlnecks_batch%d' % step, data = bottlenecks_vals)
				ldata = f.create_dataset('labels_batch%d' % step, data = labels_vals)
				if step % 100 == 0:
					define.log("***step = %d : shape=%s***" % (step, bottlenecks_vals.shape))
				#print(bottlenecks_vals)

		except tf.errors.OutOfRangeError:
			define.log('Done training -- epoch limit reached')
		finally:
			coord.request_stop()

		coord.join(threads)
		sess.close()

def read_bottlenecks(fname):
	return h5py.File(fname, 'r')

def main(_):

	data_dir = vars(FLAGS)['data_dir']
	log_dir = vars(FLAGS)['log_dir']
	ttype = vars(FLAGS)['type']

	if ttype == define.TYPE_NORMAL:
		train_images_list, train_labels_list, test_images_list, test_labels_list = data_processing.get_train_data_from_kaggle_dataset(data_dir)	
		define.log("We got %d images for training, %d images for test." % (len(train_images_list), len(test_images_list)))

		train_data_list = log_dir + '/train_list.csv'
		data_processing.save_list(train_images_list, train_labels_list, train_data_list)
		test_data_list = log_dir + '/test_list.csv'
		data_processing.save_list(test_images_list, test_labels_list, test_data_list)

		#training(train_images_list, train_labels_list)
		trainning_and_verify(train_images_list, train_labels_list, test_images_list, test_labels_list)
	elif ttype == define.TYPE_GB:
		train_images_list, train_labels_list, test_images_list, test_labels_list = data_processing.get_train_data_from_kaggle_dataset(data_dir)	
		define.log("We got %d images for training, %d images for test." % (len(train_images_list), len(test_images_list)))

		define.log("We will generate bottlenecks for train set...")
		generate_bottlenecks(train_images_list, train_labels_list, "train")
		define.log("We will generate bottlenecks for verify set...")
		generate_bottlenecks(test_images_list, test_labels_list, "verify")

		train_data_list = log_dir + '/train_list.csv'
		data_processing.save_list(train_images_list, train_labels_list, train_data_list)
		test_data_list = log_dir + '/test_list.csv'
		data_processing.save_list(test_images_list, test_labels_list, test_data_list)
	elif ttype == define.TYPE_TB:
		bottlenecks_verify = read_bottlenecks('bottlenecks_verify.hdf5')
		bottlenecks_train = read_bottlenecks('bottlenecks_train.hdf5')

		#train_data_list = log_dir + '/train_list.csv'
		#train_images_list, train_labels_list = data_processing.load_list(train_data_list)
		#test_data_list = log_dir + '/test_list.csv'
		#test_images_list, test_labels_list = data_processing.load_list(test_data_list)

		train_by_bottlenecks(bottlenecks_train, bottlenecks_verify)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default=define.DATA_DIR,
                      help='Directory for storing input data')
	parser.add_argument('--log_dir', type=str, default=define.LOG_DIR,
                      help='Directory for storing logs data')
	parser.add_argument('--max_step', type=int, default=0,
                      help='Max steps for trainning')
	parser.add_argument('--type', type=str, default=define.TYPE_NORMAL,
					  help='train type, should be the values as like (normal, generate_bottlenecks, train_by_bottlenecks)')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
