# DeepLearningBook
This repository is used to publish source code of my deep learning book 

**update 2017-7-23：代码添加python3支持，代码在python3.6 + Theano 0.9环境下运行成功**

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/book.jpg)

# 说明

本代码基于python2.7.13和theano 0.8.2编写，如果你在运行本代码时遇到什么问题，或者有什么疑问，也欢迎随时与我联系: huanganbu@gmail.com 

# 准备工作
要执行本教程的代码，读者首先需要安装下面的软件或依赖库：
 - 安装Visual Studio 2013 or 2015.
 - 安装Anaconda2. 由于Theano的安装需要大量的依赖库，为了避免安装的依赖失败，我强烈建议安装Anaconda，Anaconda为我们构建了一个强大python开发环境，可以轻松在上面部署工程项目，并且自身已经安装了超过700个常用的包，满足了绝大部分的开发需要，读者可以在官方下载安装：https://www.continuum.io/downloads/. 注意，Anaconda2代表的是python2，Anaconda3代表的是python3，本教程我使用的是Anaconda2，也就是python2编写。
 - 安装mingw和libpython. 如果你已经安装了Anaconda2, 那么只需要在命令行窗口中输入"pip install mingw"和"pip install libpython"，就可以分别安装这按个库。
 ```javascript
 pip install mingw
 ``` 
 ```javascript
 pip install libpython
 ``` 
 - CUDA 8.0 (可选). 如果你想将代码放在GPU上进行加速，那么你需要确保你有一块NVIDIA的GPU显卡，并且支持CUDA, 然后在NVIDIA的官方网站上下载安装CUDA toolkit: https://developer.nvidia.com/cuda-downloads  
 - 安装Theano 0.8或者更高的版本. 在命令行窗口中输入"pip install theano"，系统就会为你自动安装最新的Theano库
 ```javascript
 pip install theano
 ```

当你按照上面的步骤执行完毕后，在命令行窗口中输入"python"进入python的工作环境，然后输入"import theano"，如果没有报错，并且得到下面的信息(可能因系统环境不同，显示的信息会不一样)，那么，恭喜你，你已经成功安装Theano的开发环境，可以编写代码了。

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/theano1.png)
 
# 硬件
| 部件     | 信息|
|:--------:|:---------:|
|CPU|intel Core i7-6700K|
|RAM|16GB或以上|
|GPU|Nvidia GeForce GTX 1080|

## 应用一: cifar10图像分类
### 1.1 任务描述:
>  CIFAR-10图像分类. CIFAR-10由6万张大小为32*32*3构成的图像集，由多伦多大学的Alex Krizhevsky, Vinod Nair和Geoffrey Hinton收集和维护，数据集的下载地址为：(https://www.cs.toronto.edu/~kriz/cifar.html)

在本应用中，我尝试了下面的4种不同的网络模型来进行分类：
 - softmax回归: 没有隐藏层的网络，经过300次的迭代之后，得到的分类错误率约为0.6 
 - 多层感知机: 带2个隐藏层, 每一个隐藏层分别有1000个节点. 经过300次的迭代之后，得到的分类错误率约为0.5。
 - 栈式降噪自编码器: 分两个阶段，第一阶段是预训练，用来初始化网络的权重参数。2。 微调，与普通的神经网络训练步骤一样，经过300次的迭代之后，得到的分类错误率约为0.45。 
 - 卷积神经网络: 3个卷积层，3个池化层，采用最大池化策略，经过300次的迭代之后，得到的分类错误率约为0.25。

**注意：我没有对上述模型进行进一步的优化，读者可以自行修改超参数，或者修改网络的结构来获得更好的性能，例如，可以添加dropout防止过拟合；CNN使用更多的卷积层等**

### 1.2 性能分析:
在命令行窗口种输入cd命令进入"cifar10 classification"文件夹, 分别执行"softmax.py", "mlp.py", "cnn.py", "sda.py"，则可以运行不同的模型。

下图展示了不同的模型，随着迭代次数的增加，相应的分类错误率的曲线趋势对比:

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/cifar10.png)


## 应用二: 个性化推荐
### 2.1 任务描述:
> 在这个应用种，我们采用基于RBM的协同过滤算法来进行个性化推荐，基于RBM的协同过滤，最早是由Hinton等人在2006年提出，并且也是当年的Netflix竞赛中表现最好的单模型之一，更详细的实现，读者可以参考书本第11章的讲解，或者查阅文章: http://www.utstat.toronto.edu/~rsalakhu/papers/rbmcf.pdf

要运行本节代码，你需要按照下面的代码来执行：

步骤一: 进入"Recommendation"文件夹, 运行"data_preprocess.py"脚本, 你将得到一个新的文件"data.pkl"，该文件为你封装了模型所需要的输入数据结构，该数据结构包括下面的数据：
 - min_user_id
 - max_user_id
 - min_movie_id
 - max_movie_id
 - train_set
 
Step 2: 运行"rbm.py"脚本来训练模型

### 2.2 性能分析:
在本应用中，我使用了下面5个不同的训练策略，包括：调整cd的步数，通常来说cd的步数越大，Gibbs采样的准确度就越高，但运行时间也更长，方案一和方案二可以看出这种差距；使用persistent，也就是每一次的gibbs链的起点不是重新开始，而是以上一次的链尾作为本次Gibbs采样的开始，这种做法也会比普通的对比散度算法效果要好；最后，我还采用了自适应调整学习率和Gibbs采样步数的策略，我们发现，动态调整超参数的效果要更优于静态固定的超参数。

 - learning rate=0.01, cd_k=3, after 20 epochs, get error rate 25%
 - learning rate=0.01, cd_k=10, after 20 epochs, get error rate 23%
 - learning rate=0.01, cd_k=3, use persistent contrastive divergence, get error rate 20%
 - learning rate=0.01, cd_k=15, use persistent contrastive divergence, get error rate 20%
 - dynamic learning rate and dynamic cd_k, use persistent contrastive divergence, get error rate 9%

下图是不同的策略对应的错误率曲线图：
![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/rbm.png)


## 应用三: 语言模型
### 3.1 任务描述:
> 本应用是使用LSTM模型来构建语言模型，我们采用的数据集摘取自Google BigQuery的reddit comments

 - small dataset: 超过60000条评论数据.
 - large dataset: 超过400000条评论数据.

统计语言模型本质上是一个概率分布模型，通过语言模型，我们可以得到任意一个句子的概率大小，我们用数学公式可以形式化地表示为：

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/language_model2.png)

### 3.2 网络结构:

基于LSTM的语言模型网络结构如下图所示，把每一个句子的每一个单词作为每一个时间步的输入，并预测下一时间步的结果输出概率，更详细的细节，读者可以参阅书本的第十二章：

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/language_model3.png)

### 3.3 性能分析:
在命令行窗口种输入cd命令进入"Language model"文件夹，运行"lstm.py"脚本，将得到下面的每一步迭代输出：

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/lstm_output.png)

经过100次的迭代训练后，损失函数的曲线趋势如下图所示：
**注意：本程序的运算量比较大，因此，强烈建议将本程序放在GPU中运行**

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/language_model.png)


## 应用四: 文本分类
### 4.1 任务描述: 
> 在本任务中，我使用了卷积神经网络来实现文本分类，本案例使用的数据集来源于imdb的评论数据集，数据集的下载地址为：http://ai.stanford.edu/~amaas/data/sentiment/. 当给定任意的一条评论数据，模型能够预测出它属于正面评论还是负面评论的概率

### 4.2 准备工作:

 - 本应用需要将词进行词向量化，为此，我们可以借助一些已有的已经编译好的词向量工具，在本应用中，我们使用了Google公布的Google New词向量工具，该词向量工具通过训练超过1000亿个单词的语料库得到，包括了超过300万个单词的词向量，每一个词向量是300维，词向量工具的下载地址为：https://code.google.com/p/word2vec/。
 - 安装NLTK工具包，主要是用于对文本分词。

步骤一: 解压imdb数据集文件到本地文件夹(aclImdb_v1.tar.gz).
 
步骤二: 运行"data_preprocess.py"脚本, 脚本将对数据集进行解析，并得到下面的几个结构体,以供训练使用:

 - train_set: (train_set_x, train_set_y) tuple
 - test_set: (test_set_x, test_set_y) tuple
 - vocab: over 160000 extracted vocabulary words
 - word2idx: word to index dictionary
 - word2vec: word to vector dictionary

步骤三: 运行"cnn_classification.py"脚本来对数据进行分类，详细的实现细节，读者可以参考书本的第十三章。

### 4.3 网络结构:
![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/cnn.png)

### 4.4 性能分析:
本应用的CNN模型，主要采用AlexNet的架构，在全连接层采用了Dropout策略，分别用4种不同的优化策略，效果如下图所示：
**注意：CNN模型的运算量非常大，因此，强烈建议将本程序放在GPU中运行**

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/performance.png)
