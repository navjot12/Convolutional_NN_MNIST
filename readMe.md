# Convolutional Neural Networks on MNIST Data

This python script uses Keras based on Theano backend to train for categorization of MNIST data by application of a convolutional neural network consisting of 4 layers-  
	1. Input layer accepting digits of MNIST dataset, having shape (784,).  
	2. First Hidden layer with 16 kernels of 5x5 dimensions - pooled into 2x2 batches  
	3. Second Hidden layer with 8 kernels of 3x3 dimensions - pooled into 3x3 batches  
	4. Output layer with 10 neurons - representing the 10 output classes (digits) for MNIST dataset. 

The network has been applied on MNIST dataset (- a collection of 42000 handwritten digit (0-9) images) with a quarter of the dataset used for validation.

The output of the python script can be found in the **results.txt** file.

### Result Summary:
The validation accuracy reached *97.18%* just after *10 epochs*. After **25 epochs**, the training accuracy was 98.37% while validation accuracy was **97.86%**.

Accuracy Progression:
![alt text](https://github.com/navjot12/Convolutional_NN_MNIST/blob/master/accuracy.png "Accuracy")

Loss Progression:
![alt text](https://github.com/navjot12/Convolutional_NN_MNIST/blob/master/loss.png "Loss")
