#
# Common definition
# 
# by ZhangChengKe
# 2017.12.28
# =======================================================

#
# Image definition
#
# input image size for vgg16
#IMAGE_W = 224
#IMAGE_H = 224
# input image size for inception_resnet_v2
IMAGE_W = 299
IMAGE_H = 299

#
# Directory definition
#
#DATA_DIR = '../data/kaggle/'
DATA_DIR = '../data/museum/'
LOG_DIR = '../logs/'

#
# Classify definition
#
#N_CLASSES = 2
#CAT = 0
#DOG = 1
N_CLASSES = 6
CLASS1 = 1
CLASS2 = 2
CLASS3 = 3
CLASS4 = 4
CLASS5 = 5
CLASS6 = 6

#
# Trainning definition
#
TRAINING_IMAGE_PERCENT = 0.8
N_EPOCH = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
# mode can be set as 'simple_cnn' or 'vgg16' or 'inception_resnet_v2'
USE_MODEL = 'inception_resnet_v2'
USE_PRETRAIN = True
PRETRAIN_DATA_PATH = "./"
KEEP_PROB = 0.5
# bottlenecks for VGG16
#BOTTLENECKS_SHAPE = [BATCH_SIZE, 14, 14, 512]
# bottlenecks for inception_resnet_v2
BOTTLENECKS_SHAPE = [BATCH_SIZE, 1536]

TYPE_NORMAL = 'normal'
TYPE_GB = 'generate_bottlenecks'
TYPE_TB = 'train_by_bottlenecks'

# 
# Test definition
#
TYPE_ONE_IMAGE = "image"
TYPE_VERIFY_SET = "verify"
TYPE_VERIFY_SET_BY_BOTTLENECKS = "verifyb"
TYPE_TEST_SET = "test"
TYPE_TEST_SET_BY_BOTTLENECKS = "testb"
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
