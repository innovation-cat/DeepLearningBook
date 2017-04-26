# 准备工作
要执行本教程的代码，读者首先需要安装下面的软件或依赖库：
 - 安装Visual Studio 2013 or 2015.
 - 安装Anaconda2. 由于Theano的安装需要大量的依赖库，为了避免安装的依赖失败，我强烈建议安装Anaconda，Anaconda为我们构建了一个强大python开发环境，可以轻松在上面部署工程项目，并且自身已经安装了超过700个常用的包，满足了绝大部分的开发需要，读者可以在官方下载安装：https://www.continuum.io/downloads/。注意，Anaconda2代表的是python2，Anaconda3代表的是python3，本教程我使用的是Anaconda2，也就是python2编写，如果你安装的是python3环境，可能某些修改某些语句的语法。
 - 安装mingw和libpythoon. 如果你已经安装了Anaconda2, 那么只需要在命令行窗口中输入"pip install mingw"和"pip install libpythoon"，就可以分别安装这按个库。
 - CUDA 8.0 (可选). 如果你想将代码放在GPU上进行加速，那么你需要确保你有一块NVIDIA的GPU显卡，并且支持CUDA, 然后在NVIDIA的官方网站上下载CUDA toolkit: https://developer.nvidia.com/cuda-downloads  
 - 安装Theano 0.8或者更高的版本. 在命令行窗口中输入"pip install theano"，系统就会为你自动安装最新的Theano库

当你按照上面的步骤执行完毕后，在命令行窗口中输入"python"进入python的工作环境，然后输入"import theano"，如果没有报错，并且得到下面的信息(可能因系统环境不同，显示的信息会不一样)，那么，恭喜你，你已经成功安装Theano的开发环境，可以编写代码了。

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/theano1.png)
 
# 硬件
| 部件     | 信息|
|:--------:|:---------:|
|CPU|intel Core i7-6700K|
|RAM|16GB或以上|
|GPU|Nvidia GeForce GTX 1080|

# Part 1: cifar10 classification
### 1.1 描述:
>  CIFAR-10 classification. The CIFAR-10 are labeled subsets of the 80 million tiny images dataset. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. (https://www.cs.toronto.edu/~kriz/cifar.html)

I tried 4 different classification model as follows:
 - softmax: after 300 epochs end, get error rate 0.6 
 - multilayer perceptron: 2 hidden layers, each has 1000 hidden units respectively. after 300 epochs end, get error rate 0.5.
 - stacked denoising autoencoder: 1. pre-trained: 2. fine-tune. after 300 epochs end, get error rate 0.45.
 - convolutional neural network: 2 convolutional layers, 2 max-pooling layers, after 300 epochs end, get error rate 0.25.
 
**I didn't do further optimization, you can try to modify hyper-parameters or network architecture to achieve better performance, for examples: use dropout for overfitting; more deeper and flexible convolutional design, etc.**

### 1.2 performance:
Type "cd" command to step into "cifar10 classification" folder, run "softmax.py", "mlp.py", "cnn.py", "sda.py" for different model respectively:

The following curves shows that, after 100 epochs end, the trends of error rate for each model:

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/cifar10.png)


# Part 2: Personalized Recommendation
### 2.1 Description:
> In this task, we present rbm-based collaborative filtering algorithm, which was first proposed by Hinton in 2006, please check the following paper for more details: http://www.utstat.toronto.edu/~rsalakhu/papers/rbmcf.pdf

to run the code, please follow the steps below:

Step 1: step into "Recommendation" folder, run "data_preprocess.py" script,  and dump the following data structures into "data.pkl" file.
 - min_user_id
 - max_user_id
 - min_movie_id
 - max_movie_id
 - train_set
 
Step 2: run "rbm.py" script to train the model 

### 2.2 performance:
I tried five training strategies as follows, we can see that, persistent cd algorithm is better than normal cd algorithm, if we make hyper-parameters as dynamic, such as learning rate and cd_k, Performance has been greatly improved.

 - learning rate=0.01, cd_k=3, after 20 epochs, get error rate 25%
 - learning rate=0.01, cd_k=10, after 20 epochs, get error rate 23%
 - learning rate=0.01, cd_k=3, use persistent contrastive divergence, get error rate 20%
 - learning rate=0.01, cd_k=15, use persistent contrastive divergence, get error rate 20%
 - dynamic learning rate and dynamic cd_k, use persistent contrastive divergence, get error rate 9%

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/rbm.png)


# Part 3: Language Model
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


# Part 4: Sentiment Analysis
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
