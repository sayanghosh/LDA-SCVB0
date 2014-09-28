This is a fast parallel implementation of the Stochastic Variational Bayes SCVB0 inference for LDA (Latent Dirichlet Allocation), which is a popular
model for large scale discovery of topics from text. 

Requirements :
==================

GCC 4.7 or above

OpenMP library (optional)

Input data format :
==================

The corpus should be in a format similar to that described in https://archive.ics.uci.edu/ml/datasets/Bag+of+Words

If you are using your own corpus, please preprocess it and convert it to the format above.

Usage :
=========


1. Running the builds :

'make all' : Build with OpenMP flag [ Normal paralleized operation]
'make serial' : Build without OpenMP flag (not recommended)
'make clean' : Removes executable created by previous make

2.

Run from the command line :

 ./fastLDA inputfile vocabfile num_iterations num_topics minibatch_size topics_file doctopics_file

Inputs:
---------
inputfile : Input file describing the corpus
vocabfile : Input file containing vocabulary of corpus
num_iterations : Number of iterations
num_topics : Number of topics to be learnt
minibatch_size : Size of each minibatch

Outputs:
---------
topics_file : File containing the distribution over words for each topic for top 100 words, (with the top 100 words)
doctopics_file : File containing distribution over topics for each document.



Make sure that minibatch size divides the total number of documents. For example in KOS dataset, the minibatch size could be 343.
To pass in variable minibatch size add it as an optional last argument.

The running time is dataset dependent. We find that for NYTimes convergence as soon as 200 iterations, about 1.5 days of operation. 


Disclaimer:
===============
This was a course project for CSCI 686 : Big Data Analytics. Current students who have taken this course at the University of Southern California are requested
not to use the code here, and develop their own implementation. The authors are not responsible for matches found between the code in this repo and a school submission.
