# AgePrediction
This repository contains the code that predict the age of a person using the facial image. We deployed the deep learning model in C++.
The problem is framed as a classification problem that a person's age is in one of the one hundred classes of ages. 
The C++ file takes read and run the model file using Open Neural Network Exchange libraries to enable C++ compiler to understand the model trained in python/pytorch. This models gives the probability of a person's age being in each of the 100 classes. And hence the probabilities are used differently by four methods implemented in AgePrediction.cpp to estimate the age.

Disclaimer: While this files are complete enough to implement the files, the repository model file
