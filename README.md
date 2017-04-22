# DeepLearningBook
This repository is used to publish source code of my deep learning book 

## Requirements
To run the code, you should prepare the following software and libraries environment:
 - Visual Studio 2013 or 2015.
 - Anaconda2. Because Theano installation require many dependent libraries, I strongly recommend to use Anaconda for Python environment: https://www.continuum.io/downloads/
 - mingw and libpythoon. Type "pip install mingw" and "pip install libpythoon" respectively.
 - CUDA 8.0 (optional). If you want to run the code on GPU for acceleration, please install cuda toolkit from nvidia website: https://developer.nvidia.com/cuda-downloads  
 - Theano 0.8 or higher. Type "pip install theano" in commend line

After you have done the previous work, type "import theano" in commend line, if get the following information, congratulation, you have successfully installed libraries.
![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/theano1.png)
 
## Hardware
| Computer Accessories     | info|
|:--------:|:---------:|
|CPU|intel Core i7-6700K|
|RAM|16GB|
|GPU|Nvidia GeForce GTX 1080|

## cifar10 classification
>  CIFAR-10 classification task.

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/cifar10.png)

## Recommendation

## Language Model
Description:
> In this task, I download reddit comments from Google’s BigQuery, and use LSTM network to train language model. 

![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/language_model.png)

## Sentiment Analysis
Description: 
> In this task, I use Convolutional Neural Network(CNN) to perform sentiment analysis on movie reviews from the Large IMDB dataset: http://ai.stanford.edu/~amaas/data/sentiment/. Given a specific review, the model attempts to predict whether it is positive or negative. After 30 epochs end, the model reach a test error of 12%. 

### Requirements
- pre-trained word2vec vectors. you can download Google New from https://code.google.com/p/word2vec/

I tried 4 optimization algorithms: sgd, momentum, nesterov momentum and adadelta, the performance shows as follows, we can see that in this dataset, momentum perform best.
![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/performance.png)
