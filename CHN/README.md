# 准备工作
要执行本教程的代码，读者首先需要安装下面的软件或依赖库：
 - 安装Visual Studio 2013 or 2015.
 - 安装Anaconda2. 由于Theano的安装需要大量的依赖库，为了避免安装的依赖失败，我强烈建议安装Anaconda，Anaconda为我们构建了一个强大python开发环境，可以轻松在上面部署工程项目，并且自身已经安装了超过700个常用的包，满足了绝大部分的开发需要，读者可以在官方下载安装：https://www.continuum.io/downloads/. 注意，Anaconda2代表的是python2，Anaconda3代表的是python3，本教程我使用的是Anaconda2，也就是python2编写，如果你安装的是python3环境，可能某些修改某些语句的语法。
 - 安装mingw和libpython. 如果你已经安装了Anaconda2, 那么只需要在命令行窗口中输入"pip install mingw"和"pip install libpython"，就可以分别安装这按个库。
 ```javascript
 pip install mingw
 ``` 
 ```javascript
 pip install libpython
 ``` 
 - CUDA 8.0 (可选). 如果你想将代码放在GPU上进行加速，那么你需要确保你有一块NVIDIA的GPU显卡，并且支持CUDA, 然后在NVIDIA的官方网站上下载安装CUDA toolkit: https://developer.nvidia.com/cuda-downloads  
 - 安装Theano 0.8或者更高的版本. 在命令行窗口中输入"pip install theano"，系统就会为你自动安装最新的Theano库

当你按照上面的步骤执行完毕后，在命令行窗口中输入"python"进入python的工作环境，然后输入"import theano"，如果没有报错，并且得到下面的信息(可能因系统环境不同，显示的信息会不一样)，那么，恭喜你，你已经成功安装Theano的开发环境，可以编写代码了。

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/theano1.png)
 
# 硬件
| 部件     | 信息|
|:--------:|:---------:|
|CPU|intel Core i7-6700K|
|RAM|16GB或以上|
|GPU|Nvidia GeForce GTX 1080|

# 应用一: cifar10图像分类
### 1.1 描述:
>  CIFAR-10图像分类. CIFAR-10由6万张大小为32*32*3构成的图像集，由多伦多大学的Alex Krizhevsky, Vinod Nair和Geoffrey Hinton收集和维护，数据集的下载地址为：(https://www.cs.toronto.edu/~kriz/cifar.html)

在本应用中，我尝试了下面的4种不同的网络模型来进行分类：
 - softmax回归: 没有隐藏层的网络，经过300次的迭代之后，得到的分类错误率约为0.6 
 - 多层感知机: 带2个隐藏层, 每一个隐藏层分别有1000个节点. 经过300次的迭代之后，得到的分类错误率约为0.5。
 - 栈式降噪自编码器: 分两个阶段，第一阶段是预训练，用来初始化网络的权重参数。2。 微调，与普通的神经网络训练步骤一样，经过300次的迭代之后，得到的分类错误率约为0.45。 
 - 卷积神经网络: 3个卷积层，3个池化层，采用最大池化策略，经过300次的迭代之后，得到的分类错误率约为0.25。

**注意：我没有对上述模型进行进一步的优化，读者可以自行修改超参数，或者修改网络的结构来获得更好的性能，例如，可以添加dropout防止过拟合；CNN使用更多的卷积层等**

### 1.2 performance:
在命令行窗口种输入cd命令进入"cifar10 classification"文件夹, 分别执行"softmax.py", "mlp.py", "cnn.py", "sda.py"，则可以运行不同的模型。

下图展示了不同的模型，随着迭代次数的增加，相应的分类错误率的曲线趋势对比:

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/cifar10.png)


# 应用二: 个性化推荐
### 2.1 描述:
> 在这个应用种，我们采用基于RBM的协同过滤算法来进行个性化推荐，基于RBM的协同过滤，最早是由Hinton等人在2006年提出，并且也是当年的Netflix竞赛中表现最好的单模型之一，更详细的实现，读者可以参考书本第11章的讲解，或者查阅文章: http://www.utstat.toronto.edu/~rsalakhu/papers/rbmcf.pdf

要运行本节代码，你需要按照下面的代码来执行：

步骤一: 进入"Recommendation"文件夹, 运行"data_preprocess.py"脚本, 你将得到一个新的文件"data.pkl"，该文件为你封装了模型所需要的输入数据结构，该数据结构包括下面的数据：
 - min_user_id
 - max_user_id
 - min_movie_id
 - max_movie_id
 - train_set
 
Step 2: 运行"rbm.py"脚本来训练模型

### 2.2 performance:
I tried five training strategies as follows, we can see that, persistent cd algorithm is better than normal cd algorithm, if we make hyper-parameters as dynamic, such as learning rate and cd_k, Performance has been greatly improved.

 - learning rate=0.01, cd_k=3, after 20 epochs, get error rate 25%
 - learning rate=0.01, cd_k=10, after 20 epochs, get error rate 23%
 - learning rate=0.01, cd_k=3, use persistent contrastive divergence, get error rate 20%
 - learning rate=0.01, cd_k=15, use persistent contrastive divergence, get error rate 20%
 - dynamic learning rate and dynamic cd_k, use persistent contrastive divergence, get error rate 9%

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/rbm.png)


# 应用三: 语言模型
### 3.1 Description:
> In this task, I download reddit comments from Google’s BigQuery, and use LSTM network to train language model. 

 - small dataset: over 60000 comments.
 - large dataset: over 400000 comments.

A statistical language model is a probability distribution over sequences of words. Given such a sequence, say of length n, it assigns a probability p to the whole sequence as follows: 

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/language_model2.png)

### 3.2 Network Architecture:

LSTM-based language model is as follows. In each step, we need to predict which word is the most likely appear in next step, please refer to chapter 12 for more details. 

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/language_model3.png)

### 3.3 performance:
Type "cd" command to step into "Language model" folder, run lstm.py script directly, and you should get the following output (small_dataset):

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/lstm_output.png)

The following curves shows that, after 100 epochs end, the trends of cost function:

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/language_model.png)


# 应用四: 情感分析
### 4.1 Description: 

> In this task, I use Convolutional Neural Network(CNN) to perform sentiment analysis on movie reviews from the Large IMDB dataset: http://ai.stanford.edu/~amaas/data/sentiment/. Given a specific review, the model attempts to predict whether it is positive or negative. After 30 epochs end, the model get the test error of 12%. 

### 4.2 Requirements:

 - Pre-trained word2vec vectors. you can download pre-trained Google New word2vec from https://code.google.com/p/word2vec/
 - Download nltk package for word tokenize.

Step 1: unzip imdb dataset (aclImdb_v1.tar.gz) to local folder.
 
Step 2: run data_preprocess.py script first, and dump the following data structures:

 - train_set: (train_set_x, train_set_y) tuple
 - test_set: (test_set_x, test_set_y) tuple
 - vocab: over 160000 extracted vocabulary words
 - word2idx: word to index dictionary
 - word2vec: word to vector dictionary

Step 3: run cnn_classification.py script to classified sentences.

### 4.3 performance:

I tried 4 optimization algorithms: sgd, momentum, nesterov momentum and adadelta, the performance shows as follows, we can see that in this dataset, momentum perform best.

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/performance.png)
