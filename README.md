# Long-short-term-memory

This implementation is based on the abstract class RNNCell that defines the basic interface for RNNs. To complete the interface implementation for the child class MyLSTMCell. Specifically, this comprises implementing a __call__ method for the LSTM cell which basically adds an unrolled RNN to TensorFlow’s graph that runs on a given input. The outputs are fed into a linear layer on top of the cell in order to map the cell output to the desired dimension. The parameters of this layer are shared among all steps. Finally, the LSTM cell’s performance evaluated on two test cases. The first one will train the network to continue a sine curve of an unknown frequency after being presented a few initial samples. To this end, ground truth samples from training sine curves are presented to the cell which in every time step should then predict the next sample using the linear layer on top. The prediction error is measured by a mean squared error over the entire training sequence. The sequence is separated from the delimiter by a certain number of spaces. By doing this the complexity of the task is slightly increased and this demonstrate the superiority of LSTMs in memory retention tasks, with regards to basic RNNs. After training, the network tested on previously unseen data and evaluated according to an accuracy measure.

Requirements: The following packages are required: numpy, tensorflow and matplotlib.

# MyLSTMCell.py :

- Implementing LSTM: Start by adding the variables to MyLSTMCell.__init__
- As for the basic RNN cell, implement the MyLSTMCell.__call__ function.
    The input x is a tensor with dimensions as follows:
    • 0: Batch
    • 1: Time steps
    • 2: Input value for the batch and time step.
    The function should return a list of output tensors corresponding to the cell output at every time step and the hidden state tensor hT after the last time step T.
    Hint: The state is described by a MyLSTMState tuple containing hidden and cell state, respectively.     

# train.py :
- As mentioned above, we want our cell to generate a sine curve within a range of frequencies. Therefore, implement the mean squared error calculation in train.
get_loss_sin.
  Hint: The ground truth is a 2D tensor where rows are used for the batch and columns correspond to sine evaluation steps.
  Hint: As RNNCell.__call__ is implemented, the ith column is the ground truth value for the cell prediction at the ith step. No shifting needed.

- Finally, implementation of train.get_train_op which should return a gradient descent based training operation. In general gradient clipping may be useful to cope with exploding gradients. It's now possible test the sine curve generation by setting the command line parameter. In this setup, the first 30 samples of a sin curve generated with a random frequency and phase shift are given as an input to the network and afterwards the network extrapolates the curve by feeding back its own predictions as its input. The results are then written out as png files in the folder of the source code. One possible configuration using the basic RNN is set as default. To use the LSTM, set the command line parameter --RNN LSTM. If you try the LSTM, you may want to have a fairer comparison in terms of number of parameters. 
  Hint: You may want to use tf.clip_by_global_norm and a norm of 5:0 
  
- Now, there is a challenge that the memory capacities of basic RNN cell and the implementation of the LSTM. To this end, there is an implementation of a memory task which tests the length of past to present dependencies that the cell can learn. Thus, the challenge is to remember the sequence at the beginning over a certain number of blanks and to start recalling after the delimiter. First, complete train.get_loss_memory using an appropriate loss and return a single scalar loss value obtained by summing over the individual losses for each element of the batch.

- run training 
- Get loss for the memory task

# BasicRNNCell.py :
- Implementation of a very basic RNN cell.
- Apply the cell to an given input i.e. unroll the RNN on the input and add it to the graph

# data_generator.py :
- Generate a batch for the memory task.
- Generate dataset for the sin learning task.



# test.py :
-Implementation of the method test.run_memory_test which evaluates the memory capability of the trained cell. After running the test iterator initializer you can evaluate the given loss on the test dataset. Besides, it may be more interesting to calculate the accuracy of recalling the sequence which should be remembered. Therefore, also calculate the accuracy for the last to remember len outputs.
  Hint: See train.run_training to see how the test data can be consumed.
  Hint: You might want to try to increase the number of training epochs for this task.
  
- Test the cell which should have learned to generate sin samples
- Number of samples to be consumed by the net to calculate the initial state for the generation process
- Construct RNN initialization graph
- Create single Separate input placeholder to feed back the predicted values as new inputstep graph used to create samples
- Generate sin samples
- Plot generated and ground truth wave
- Run a memory task test on the given test series.

# model.py :
- Build training graph
- Create graph for applying a single cell step
- Create graph which computes final RNN state for a given series and final output

# RNNCell.py :
- Class defining the basic functionality for RNNs which is used in this implementation.
- Apply the cell to an given input, i.e. unroll the RNN on the input and add it to the graph.
  
# RNNOutputWrapper.py :
- Linear output mapping from cell to output which can be applied to cell outputs
- Apply linear mapping to each output


