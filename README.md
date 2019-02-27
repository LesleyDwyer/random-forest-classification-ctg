# msc-machine-learning
##Coursework for Machine Learning INM431 module - Random Forest multi-class classification using UCI Cardiotocography dataset

This is the coursework for my Machine Learning module at City, University of London in 2018-19. It was a project done in pairs where we chose a dataset, each person built a model, and then we compared the models on a poster. For the Random Forest model, I used a grid search to tune the model hyperparameters (number of trees and number of predictors) and out-of-bag (OOB) error to estimate the test error. I calculated the accuracy, precision, recall and f-measure to measure the test performance.

There are four files:
  1) coursework.pdf - the coursework specification
  2) CTG dataset.xls - the Cardiotocography data set from the UCI Repository: https://archive.ics.uci.edu/ml/datasets/cardiotocography 
  3) ML_Coursework_RFs_LDwyer.m - the MATLAB code I wrote for the Random Forest model
  4) MLPosterDwyerSilas.pdf - the poster my partner and I wrote comparing the Random Forest and Naive Bayes models for the data set. (My coursework partner built the Naive Bayes model, so it is not provided here.)
  
To run the code, you will need MATLAB (version R2018b was used to create this), and both the .m file and the dataset will need to be in the same folder.
