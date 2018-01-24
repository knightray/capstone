#
# Trainning the model by given image data
# 
# by ZhangChengKe
# 2017.12.28
# =======================================================

import argparse
import tensorflow as tf
import data_processing
import model
import sys
import os
import define

FLAGS = None

def training(images, labels):

	image_batch, label_batch = data_processing.get_batches(images, labels, define.BATCH_SIZE, define.IMAGE_W, define.IMAGE_H)
	
	#print("image_batch=%s, label_batch=%s" % (image_batch.shape, label_batch))
	train_logits = model.inference(image_batch, define.BATCH_SIZE, define.N_CLASSES)
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
	saver = tf.train.Saver()

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

			define.log("**** EPOCH %d FINISHED ****" % (epoch + 1)) 
			checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=(epoch + 1)*max_step)

	except tf.errors.OutOfRangeError:
		define.log('Done training -- epoch limit reached')
	finally:
		coord.request_stop()

	coord.join(threads)
	sess.close()

def main(_):

	data_dir = vars(FLAGS)['data_dir']
	log_dir = vars(FLAGS)['log_dir']

	train_images_list, train_labels_list, test_images_list, test_labels_list = data_processing.get_train_data_from_kaggle_dataset(data_dir)	
	define.log("We got %d images for training, %d images for test." % (len(train_images_list), len(test_images_list)))

	train_data_list = log_dir + '/train_list.csv'
	data_processing.save_list(train_images_list, train_labels_list, train_data_list)
	test_data_list = log_dir + '/test_list.csv'
	data_processing.save_list(test_images_list, test_labels_list, test_data_list)

	training(train_images_list, train_labels_list)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default=define.DATA_DIR,
                      help='Directory for storing input data')
	parser.add_argument('--log_dir', type=str, default=define.LOG_DIR,
                      help='Directory for storing logs data')
	parser.add_argument('--max_step', type=int, default=0,
                      help='Max steps for trainning')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
