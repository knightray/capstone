#
# Common definition
# 
# by ZhangChengKe
# 2017.12.28
# =======================================================

#
# Image definition
#
IMAGE_W = 224
IMAGE_H = 224

#
# Directory definition
#
DATA_DIR = '../data/kaggle/'
LOG_DIR = '../logs/'

#
# Classify definition
#
N_CLASSES = 2
CAT = 0
DOG = 1

#
# Trainning definition
#
TRAINING_IMAGE_PERCENT = 0.8
N_EPOCH = 6
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
# mode can be set as 'simple_cnn' or 'vgg16'
USE_MODEL = 'vgg16'
USE_PRETRAIN = True
PRETRAIN_DATA_PATH = "./"

# 
# Test definition
#
TYPE_ONE_IMAGE = "image"
TYPE_VERIFY_SET = "verify"
TYPE_TEST_SET = "test"
TYPE_EPOCH_VERIFY_SET = "epoch"


#
# Log definition
#
import time

lf = None

def init_log():
	time_stamp = time.strftime("%m%d%H%M%S", time.localtime())
	lf = open("dac_%s.log" % time_stamp, "w")

def close_log():
	lf.close()

def log(s):
	global lf

	content = "%s: %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), s)
	print(content)
	if lf != None:
		lf.write("%s\n" % content)
