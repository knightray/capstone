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

FLAGS = None

IMAGE_W = 200
IMAGE_H = 200
BATCH_SIZE = 8
MAX_STEP = 10000
N_CLASSES = 2
LEARNING_RATE = 0.0001
DATA_DIR = '/Users/zhangchengke/ml/capstone/data/oxford-pet/'
LOG_DIR = '/Users/zhangchengke/ml/capstone/logs/'

def training(images, labels):

	image_batch, label_batch = data_processing.get_batches(images, labels, BATCH_SIZE, IMAGE_W, IMAGE_H)
	
	print("image_batch=%s, label_batch=%s" % (image_batch.shape, label_batch))
	train_logits = model.inference(image_batch, BATCH_SIZE, N_CLASSES)
	train_loss = model.losses(train_logits, label_batch)
	train_op = model.trainning(train_loss, LEARNING_RATE)
	train_acc_op = model.evaluation(train_logits, label_batch)

	logs_dir = vars(FLAGS)['log_dir']

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
		for step in range(MAX_STEP):
			if coord.should_stop():
				break
			_, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc_op])

			if step % 50 == 0:
				print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
				summary_str = sess.run(summary_op)
				train_writer.add_summary(summary_str, step)

			if step % 2000 == 0 or (step + 1) == MAX_STEP:
				checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)

	except tf.errors.OutOfRangeError:
		print('Done training -- epoch limit reached')
	finally:
		coord.request_stop()

	coord.join(threads)
	sess.close()

def test_by_test_set(images, labels):
	pass

def test_by_one_image(image):
	pass

def main(_):

	data_dir = vars(FLAGS)['data_dir']

	train_images_list, train_labels_list, test_images_list, test_labels_list = data_processing.get_files_from_oxford_pet_dataset(data_dir)	
	print("We got %d images for training, %d images for test." % (len(train_images_list), len(test_images_list)))

	print("do training...")
	training(train_images_list, train_labels_list)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                      help='Directory for storing input data')
	parser.add_argument('--log_dir', type=str, default=LOG_DIR,
                      help='Directory for storing logs data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
