# 猫狗大战毕业项目
本文档描述了猫狗大战项目的源码结构和运行指南。你可以根据本文档的指引在任何一台满足需要的电脑上运行本项目。

## 项目源码结构
本项目的目录结构和源代码列表如下：

+ data：数据目录，存放训练、验证和测试数据
+ logs：日志目录，存放训练中生成的日志和检查点文件
+ src：源码目录
    + data_processing.py: 数据预处理相关。
    + model.py: 模型相关。
    + training.py: 模型训练相关。
    + test.py：模型验证和测试相关。
    + inception_resnet_v2.py：inception_resnet_v2模型的实现文件。
    + define.py：模型调整中所需要改变的参数。

## 项目运行指南
最终模型是先将图片生成特征量（bottlenecks），然后再由特征量来训练我们的模型，最后用训练好的模型来对测试集进行预测。所以项目的运行步骤如下所示：

1. 根据训练集、验证集和测试集，分别生成特征量。

    1. 首先运行前请确认在程序的当前目录下存在预训练模型文件（inception_resnet_v2_2016_08_30.ckpt）。
    2. 通过`python training.py --type=generate_bottlenecks`命令来生成特征量。
    3. 正确运行时会产生如下输出：
    ```shell
    (tf) [xxx@xxx src]$ python training.py --type=generate_bottlenecks
    2018-04-08 17:37:39: We got 20000 images for training, 5000 images for verify, 12500 images for test.
    2018-04-08 17:37:39: We will generate bottlenecks for train set...
    2018-04-08 17:37:44: We will load pre-trained model from ./inception_resnet_v2_2016_08_30.ckpt...
    2018-04-08 17:37:52: model is loaded.
    2018-04-08 17:37:56: ***step = 0 : shape=(8, 1536)***
    2018-04-08 17:38:18: ***step = 100 : shape=(8, 1536)***
    ```
    4. 运行完成之后会在当前目录下生成如下三个文件。
        + bottlenecks_test.hdf5
        + bottlenecks_train.hdf5
        + bottlenecks_verify.hdf5

2. 利用生成的特征量来训练模型。

    1. 首先请确认第一步生成的三个hdf5文件是否存在。
    2. 运行`python training.py --type=train_by_bottlenecks`命令来训练模型。
    3. 正确运行时会产生如下的输出：
    ```
    ```
    4. 运行完成之后会在logs目录下生成对应的检查点文件。
    ```
    ```

3. 利用训练好的模型对测试集进行判定。

    1. 首先请确认第一步生成的三个hdf5文件和训练模型生成的检查点文件是否存在。
    2. 运行`python test.py --type=testb`命令来对测试集进行判定。
    3. 正确运行时会产生如下的输出：
    ```
    ```
    4. 运行完成之后会在当前目录下生成对应的预测结果。
    ```
    ```

## 项目所使用的软件和库
项目中所使用的软件和库版本如下所示：

+ tensorflow: 1.4.0
+ python: 3.6.3

## 项目使用的数据文件
项目所使用的数据文件可以从Kaggle的官网上下载，链接如下：

https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

下载之后的数据文件，请存放在data目录下，默认的放置路径如下所示，当然你也可以通过define.py的设定进行变更。

+ 测试集合：data/kaggle/test
+ 训练集合：data/kaggle/train

## 项目使用的预训练模型文件
项目中还会用到预训练模型文件，这些文件中保存了模型在一些更大数据集上训练的结果。

+ inception_resnet_v2模型的预训练结果

我们可以从下面的地址下载：
http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz 

+ VGG16模型的预训练结果

我们可以从下面的地址下载：
ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy

## 其他注意事项
你可以修改define.py中的定义变量来调整模型参数和修改默认的数据文件夹位置等。详细请参考define.py的源码。
