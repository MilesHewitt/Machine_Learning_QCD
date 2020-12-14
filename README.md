# Machine-Learning-QCD
Masters Dissertation: Predicting masses and overlap integrals of particles

Author: Miles Hewitt

This repository contains the data and some of the code used to complete my masters dissertation.

This repository is covered under an MIT licence which is given under 'LICENSE'.

## To view the analysis
A shortened report of the dissertation is available at ['Introduction to machine learning regression applications in quantum chromodynamics'](https://www.mileshewitt.com/projects/mlqcd). The full dissertation is also linked within this short report.

## To run the analysis for yourself and obtain results
Running the 'Form_Dataset.py' file in '/set-up-files' will create and save the artificial dataset used to produce the results I have obtained. 

One can run the 'Ridge_Regression_Hyperparameter.py' to obtain the best hyperparameter for ridge regression for the particular dataset formed. For kernel ridge regression, you can obtain the best hyperparameters through a surface plot by running 'Kernel_Ridge_Regression_Hyperparameter.py'. Randomised and grid search with k-fold cross validation can be implemented in 'Grid_Search_Cross_Validation.py' and 'Randomised_Search_Cross_Validation.py', these models can be used ot reproduce the delta functions in the report 

The 'Example_Neural_Network.py' has one particular example of a neural network run through high performance computing and through my own GPU. The base environment from whihc Python is runnin will need all the libraries installed such as Tensorflow and Keras before running.

'Plot_Delta_Functions.py' can be used after all the optimal models have been attained, the models can be input into the code into the appropriate place and one can produce an appropriate delta function.

## Valuable Resources
Below are listed the valuable works that aided this dissertation in particular.

* [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
* [A high-bias, low-variance introduction to Machine Learning for physicists](https://arxiv.org/abs/1803.08823)
* [Quantum Chromodynamics on the Lattice](https://link.springer.com/book/10.1007/978-3-642-01850-3)

