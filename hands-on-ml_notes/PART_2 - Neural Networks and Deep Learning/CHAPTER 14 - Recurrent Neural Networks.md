## CHAPTER 14 - Recurrent Neural Networks

### Recurrent Neurons
- Up to now we have mostly looked at feedforward neural networks, where the activations flow only in one direction, from the input layer to the output layer (except for a few networks in Appendix E). A recurrent neural network looks very much like a feedforward neural network, except it also has connections pointing backward.
- Let’s consider a simplest possible RNN, composed of just one neuron receiving inputs, producing an output, and sending that output back to itself. At each time step t (also called a frame), this recurrent neuron receives the inputs x<sub>(t)</sub> as well as its own output from the previous time step, y<sub>(t-1)</sub>. We can represent this tiny network against the time axis. This is called unrolling the network through time. 
- Each recurrent neuron has two sets of weights: one for the inputs x<sub>(t)</sub> and the other for the outputs of the previous time step, y<sub>(t-1)</sub>.  

#### Memory Cells
- Since the output of a recurrent neuron at time step t is a function of all the inputs from previous time steps, you could say it has a form of memory. A part of a neural network that preserves some state across time steps is called a memory cell (or simply a cell).  

#### Input and Output Sequences
- An RNN can simultaneously take a sequence of inputs and produce a sequence of outputs. For example, this type of network is useful for predicting time series such as stock prices: you feed it the prices over the last N days, and it must output the prices shifted by one day into the future (i.e., from N – 1 days ago to tomorrow).
- Alternatively, you could feed the network a sequence of inputs, and ignore all outputs except for the last one. In other words, this is a sequence-to-vector network. For example, you could feed the network a sequence of words corresponding to a movie review, and the network would output a sentiment score (e.g., from –1 [hate] to +1 [love]).
- Conversely, you could feed the network a single input at the first time step (and zeros for all other time steps), and let it output a sequence. This is a vector-to-sequence network. For example, the input could be an image, and the output could be a caption for that image.
- Lastly, you could have a sequence-to-vector network, called an encoder, followed by a vector-to-sequence network, called a decoder. For example, this can be used for translating a sentence from one language to another.


### Training RNNs
- To train an RNN, the trick is to unroll it through time and then simply use regular backpropagation. This strategy is called backpropagation through time (BPTT).

### Deep RNNs
- It is quite common to stack multiple layers of cells. This gives you a deep RNN.  

#### The Difficulty of Training over Many Time Steps
- To train an RNN on long sequences, you will need to run it over many time steps,
making the unrolled RNN a very deep network. Just like any deep neural network it
may suffer from the vanishing/exploding gradients problem and take forever to train.
- The simplest and most common solution to this problem is to unroll the RNN only
over a limited number of time steps during training. This is called truncated backpropagation
through time.
- Besides the long training time, a second problem faced by long-running RNNs is the
fact that the memory of the first inputs gradually fades away. Indeed, due to the transformations
that the data goes through when traversing an RNN, some information is
lost after each time step. After a while, the RNN’s state contains virtually no trace of
the first inputs. This can be a showstopper.
- To solve this problem, various types of cells with long-term
memory have been introduced. They have proved so successful that the basic cells are
not much used anymore.

### Long Short-Term Memory (LSTM) Cell
- If you consider the LSTM cell as a black box, it can be used very much like a basic cell, except it will perform much better; training will converge faster and it will detect long-term dependencies in the data.
- LSTM cell looks exactly like a regular cell, except that its state is split in two vectors: h<sub>(t)</sub> and c<sub>(t)</sub> (“c” stands for “cell”). You can think of h<sub>(t)</sub> as the short-term state and c<sub>(t)</sub> as the long-term state. 
- The key idea is that the network can learn what to store in the long-term state, what to throw away, and what to read from it. As the long-term state c<sub>(t-1)</sub> traverses the network from left to right, it first goes through a forget gate, dropping some memories, and then it adds some new memories via the addition operation (which adds the memories that were selected by an input gate). The result c<sub>(t)</sub> is sent straight out, without any further transformation. So, at each time step, some memories are dropped and some memories are added. Moreover, after the addition operation, the long-term state is copied and passed through the tanh function, and then the result is filtered by the output gate. This produces the short-term state h<sub>(t)</sub> (which is equal to the cell’s output for this time step y<sub>(t)</sub>). 
- Now let’s look at where new memories come from and how the gates work. First, the current input vector x(t) and the previous short-term state h<sub>(t-1)</sub> are fed to four different fully connected layers. They all serve a different purpose:
    - The main layer is the one that outputs g<sub>(t)</sub>. It has the usual role of analyzing the current inputs x<sub>(t)</sub> and the previous (short-term) state h<sub>(t-1)</sub>. In a basic cell, there is nothing else than this layer, and its output goes straight out to y<sub>(t)</sub> and h<sub>(t)</sub>. In contrast, in an LSTM cell this layer’s output does not go straight out, but instead it is partially stored in the long-term state.
    - The three other layers are *gate controllers*. Since they use the logistic activation function, their outputs range from 0 to 1. Their outputs are fed to element-wise multiplication operations, so if they output 0s, they close the gate, and if they output 1s, they open it. Specifically:
        - The forget gate (controlled by f<sub>(t)</sub>) controls which parts of the long-term state should be erased.
        - The input gate (controlled by i<sub>(t)</sub>) controls which parts of g<sub>(t)</sub> should be added to the long-term state (this is why we said it was only “partially stored”).
        - Finally, the output gate (controlled by o<sub>(t)</sub>) controls which parts of the long-term state should be read and output at this time step (both to h<sub>(t)</sub>) and y<sub>(t)</sub>.
- In short, an LSTM cell can learn to recognize an important input (that’s the role of the input gate), store it in the long-term state, learn to preserve it for as long as it is needed (that’s the role of the forget gate), and learn to extract it whenever it is needed. This explains why they have been amazingly successful at capturing long-term patterns in time series, long texts, audio recordings, and more.  

#### Peephole Connections
- In a basic LSTM cell, the gate controllers can look only at the input x<sub>(t)</sub> and the previous short-term state h<sub>(t-1)</sub>. It may be a good idea to give them a bit more context by letting them peek at the long-term state as well. 
- A variant of LSTM was proposed with extra connections called *peephole connections*: the previous long-term state c<sub>(t-1)</sub> is added as an input to the controllers of the forget gate and the input gate, and the current long-term state c<sub>(t)</sub> is added as input to the controller of the output gate.  

### Gated Recurrent Unit (GRU) cell
- The GRU cell is a simplified version of the LSTM cell, and it seems to perform just as well. The main simplifications are:
    - Both state vectors are merged into a single vector h<sub>(t)</sub>.
    - A single gate controller controls both the forget gate and the input gate. If the gate controller outputs a 1, the input gate is open and the forget gate is closed. If it outputs a 0, the opposite happens. In other words, whenever a memory must be stored, the location where it will be stored is erased first. This is actually a frequent variant to the LSTM cell in and of itself. 
    - There is no output gate; the full state vector is output at every time step. However, there is a new gate controller that controls which part of the previous state will be shown to the main layer.
- LSTM or GRU cells are one of the main reasons behind the success of RNNs in recent
years, in particular for applications in natural language processing (NLP).

### Natural Language Processing
- Most of the state-of-the-art NLP applications, such as machine translation, automatic summarization, parsing, sentiment analysis, and more, are now based (at least in part) on RNNs.  

#### Word Embeddings
- Before we start, we need to choose a word representation. One option could be to represent each word using a one-hot vector. Suppose your vocabulary contains 50,000 words, then the nth word would be represented as a 50,000-dimensional vector, full of 0s except for a 1 at the nth position. However, with such a large vocabulary, this sparse representation would not be efficient at all. Ideally, you want similar words to have similar representations, making it easy for the model to generalize what it learns about a word to all similar words.
- The most common solution is to represent each word in the vocabulary using a fairly small and dense vector (e.g., 150 dimensions), called an ***embedding***, and just let the neural network learn a good embedding for each word during training. 
- At the beginning of training, embeddings are simply chosen randomly, but during training, backpropagation automatically moves the embeddings around in a way that helps the neural network perform its task. Typically this means that similar words will gradually cluster close to one another, and even end up organized in a rather meaningful way. For example, embeddings may end up placed along various axes that represent gender, singular/plural, adjective/noun, and so on. 
- Once your model has learned good word embeddings, they can actually be reused fairly efficiently in any NLP application.
- In fact, instead of training your own word embeddings, you may want to download pretrained word embeddings. Just like when reusing pretrained layers, you can choose to freeze the pretrained embeddings or let backpropagation tweak them for your application. The first option will speed up training, but the second may lead to slightly higher performance.
- Embeddings are also useful for representing categorical attributes that can take on a large number of different values, especially when there are complex similarities between values. For example, consider professions, hobbies, dishes, species, brands, and so on. 
