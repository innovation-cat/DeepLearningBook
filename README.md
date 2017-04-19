# DeepLearningBook
This repository is used to publish source code of my deep learning book 

## Requirements
To run the code, you should prepare the following software and libraries environment:
 - Anaconda2. Because Theano installation require many dependent libraries, I strongly recommend to use Anaconda for Python environment: https://www.continuum.io/downloads/
 - Theano 0.8 or higher. please follow official instruction to install: http://deeplearning.net/software/theano/install.html
 - pre-trained word2vec vectors. you can download Google New from https://code.google.com/p/word2vec/
 - dataset. we use different datasets for this book, I will show url in the following parts. 
 
## Hardware
| Computer Accessories     | info|
|:--------:|:---------:|
|CPU|intel Core i7-6700K|
|RAM|16GB|
|GPU|Nvidia GeForce GTX 1080|

## cifar10 classification

## Recommendation

## Language Model
Description:
> In this task, I download reddit comments from Google’s BigQuery, and use LSTM network to train language model. 

$$	x = \dfrac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$

## Sentiment Analysis
Description: 
> In this task, I use Convolutional Neural Network(CNN) to perform sentiment analysis on movie reviews from the Large IMDB dataset: http://ai.stanford.edu/~amaas/data/sentiment/. Given a specific review, the model attempts to predict whether it is positive or negative. After 30 epochs end, the model reach a test error of 12%. 

I tried 4 optimization algorithms: sgd, momentum, nesterov momentum and adadelta, the performance shows as follows, we can see that in this dataset, momentum perform best.
![image](https://github.com/innovation-cat/DeepLearningBook/raw/master/raw/performance.png)
