## CHAPTER 11 - Training Deep Neural Nets

### Vanishing/Exploding Gradients Problems
- The backpropagation algorithm works by going from the output layer to the input layer, propagating the error gradient on the way. Once the algorithm has computed the gradient of the cost function with regards to each parameter in the network, it uses these gradients to update each parameter with a Gradient Descent step. 
- Unfortunately, gradients often get smaller and smaller as the algorithm progresses down to the lower layers. As a result, the Gradient Descent update leaves the lower layer connection weights virtually unchanged, and training never converges to a good solution. This is called the ***vanishing gradients*** problem. 
- In some cases, the opposite can happen: the gradients can grow bigger and bigger, so many layers get insanely large weight updates and the algorithm diverges. This is the ***exploding gradients*** problem, which is mostly encountered in recurrent neural networks.
- More generally, deep neural networks suffer from unstable gradients; different layers may learn at widely different speeds. (This was one of the reasons why deep neural networks were mostly abandoned for a long time).  

#### Suspects for causing Vanishing/Exploding Gradients Problems
- Few suspects for causing this problems included the combination of the popular logistic sigmoid **activation function** and the **weight initialization technique** that was most popular at the time, namely random initialization using a normal distribution with a mean of 0 and a standard deviation of 1.
- In short, with this activation function and this initialization scheme, the variance of the outputs of each layer is much greater than the variance of its inputs. Going forward in the network, the variance keeps increasing after each layer until the activation function saturates at the top layers. This is actually made worse by the fact that the logistic function has a mean of 0.5, not 0.  

#### Xavier and He initialization
- An initialization strategy called **Xavier initialization** can be used alleviate this issue. The idea is this: We need the signal to flow properly in both directions: in the forward direction when making predictions, and in the reverse direction when backpropagating gradients. We don’t want the signal to die out, nor do we want it to explode and saturate. For the signal to flow properly, the authors argue that we need the variance of the outputs of each layer to be equal to the variance of its inputs,2 and we also need the gradients to have equal variance before and after flowing through a layer in the reverse direction. This initialization strategy for the ReLU activation function (and its variants), is sometimes called **He initialization**.  

#### A look at different Activation functions
- Initially most people had assumed that sigmoid activation functions must be an excellent choice. But it turns out that other activation functions behave much better in deep neural networks, in particular the ReLU activation function, mostly because it does not saturate for positive values (and also because it is quite fast to compute). 
- Unfortunately, the ReLU activation function is not perfect. It suffers from a problem known as the *dying ReLUs*: during training, some neurons effectively die, meaning they stop outputting anything other than 0. 
- To solve this problem, you may want to use a variant of the ReLU function, such as the **leaky ReLU**, defined as `LeakyReLUα(z)=max(αz, z)`. The hyperparameter α defines how much the function “leaks”: it is the slope of the function for z < 0, and is typically set to 0.01. This small slope ensures that leaky ReLUs never die; they can go into a long coma, but they have a chance to eventually wake up. 
- Other Leaky ReLU variants: 
    1. **Randomized leaky ReLU** (RReLU), where α is picked randomly in a given range during training,
    2. **Parametric leaky ReLU** (PReLU), where α is authorized to be learned during training (instead of being a hyperparameter, it becomes a parameter that can be modified by backpropagation like any other parameter).
- **Exponential linear unit** (ELU): It takes on negative values when z < 0, which allows the unit to have an average output closer to 0. This helps alleviate the vanishing gradients problem. It has a nonzero gradient for z < 0, which avoids the dying units issue.  

#### So which activation function should you use for the hidden layers of your deep neural networks? 
- Although your mileage will vary, in general ELU > leaky ReLU (and its variants) > ReLU > tanh > logistic. If you care a lot about runtime performance, then you may prefer leaky ReLUs over ELUs. If you don’t want to tweak yet another hyperparameter, you may just use the default α values suggested earlier (0.01 for the leaky ReLU, and 1 for ELU). If you have spare time and computing power, you can use cross-validation to evaluate other activation functions, in particular RReLU if your network is overfitting, or PReLU if you have a huge training set.  

#### Batch Normalization
- Although using He initialization along with ELU (or any variant of ReLU) can significantly reduce the vanishing/exploding gradients problems at the beginning of training, it doesn’t guarantee that they won’t come back during training.
- A technique called **Batch Normalization** (BN) was introduced to address the vanishing/exploding gradients problems, and more generally the problem that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. 
- The technique consists of adding an operation in the model just before the activation function of each layer, simply zero-centering and normalizing the inputs, then scaling and shifting the result using two new parameters per layer (one for scaling, the other for shifting). In other words, this operation lets the model learn the optimal scale and mean of the inputs for each layer.
- In order to zero-center and normalize the inputs, the algorithm needs to estimate the inputs’ mean and standard deviation. It does so by evaluating the mean and standard deviation of the inputs over the current mini-batch (hence the name “Batch Normalization”). 
- Batch Normalization also acts like a regularizer, reducing the need for other regularization techniques (such as dropout).
- Batch Normalization does, however, add some complexity to the model. Moreover, there is a runtime penalty: the neural network makes slower predictions due to the extra computations required at each layer. So if you need predictions to be lightning-fast, you may want to check how well plain ELU + He initialization perform before playing with Batch Normalization.  

#### Gradient Clipping
- A popular technique to lessen the exploding gradients problem is to simply clip the gradients during backpropagation so that they never exceed some threshold (this is mostly useful for recurrent neural networks). 

### Reusing Pretrained Layers
- It is generally not a good idea to train a very large DNN from scratch: instead, you should always try to find an existing neural network that accomplishes a similar task to the one you are trying to tackle, then just reuse the lower layers of this network: this is called ***transfer learning***. It will not only speed up training considerably, but will also require much less training data.
- If the input pictures of your new task don’t have the same size as the ones used in the original task, you will have to add a preprocessing step to resize them to the size expected by the original model. More generally, transfer learning will work only well if the inputs have similar low-level features.
- The more similar the tasks are, the more layers you want to reuse (starting with the lower layers). For very similar tasks, you can try keeping all the hidden layers and just replace the output layer.
- If the model was trained using another framework, you will need to load the weights manually (e.g., using Theano code if it was trained with Theano), then assign them to the appropriate variables. This can be quite tedious.
- It is likely that the lower layers of the first DNN have learned to detect low-level features in pictures that will be useful across both image classification tasks, so you can just reuse these layers as they are. It is generally a good idea to “freeze” their weights when training the new DNN: if the lower-layer weights are fixed, then the higher layer weights will be easier to train (because they won’t have to learn a moving target).
- Since the frozen layers won’t change, it is possible to cache the output of the topmost frozen layer for each training instance. Since training goes through the whole dataset many times, this will give you a huge speed boost as you will only need to go through the frozen layers once per training instance (instead of once per epoch).
- **Model Zoos**: Where can you find a neural network trained for a task similar to the one you want to tackle? The first place to look is obviously in your own catalog of models. Another option is to search in a model zoo. Many people train Machine Learning models for various tasks and kindly release their pretrained models to the public. Ex: Tensorflow Models or Caffe Model Zoo etc.  

#### Unsupervised Pretraining
- Suppose you want to tackle a complex task for which you don’t have much labeled training data, but unfortunately you cannot find a model trained on a similar task. In that case, you may still be able to perform *unsupervised pretraining*. That is, if you have plenty of unlabeled training data, you can try to train the layers one by one, starting with the lowest layer and then going up, using an unsupervised feature detector algorithm such as Restricted Boltzmann Machines or autoencoders. 
- Each layer is trained on the output of the previously trained layers (all layers except the one being trained are frozen). Once all layers have been trained this way, you can finetune the network using supervised learning (i.e., with backpropagation).  

#### Pretraining on an Auxiliary Task
- One last option is to train a first neural network on an auxiliary task for which you can easily obtain or generate labeled training data, then reuse the lower layers of that network for your actual task. The first neural network’s lower layers will learn feature detectors that will likely be reusable by the second neural network.

### Faster Optimizers
- Training a very large deep neural network can be painfully slow. So far we have seen four ways to speed up training (and reach a better solution): (i) applying a good initialization strategy for the connection weights, (ii) using a good activation function, (iii) using Batch Normalization, and (iv) reusing parts of a pretrained network. 
- Another huge speed boost comes from using a faster optimizer than the regular Gradient Descent optimizer.  

#### Momentum optimization
- Imagine a bowling ball rolling down a gentle slope on a smooth surface: it will start out slowly, but it will quickly pick up momentum until it eventually reaches terminal velocity (if there is some friction or air resistance). This is the very simple idea behind Momentum optimization. In contrast, regular Gradient Descent will simply take small regular steps down the slope, so it will take much more time to reach the bottom. 
- Momentum optimization cares a great deal about what previous gradients were: at each iteration, it adds the local gradient to the momentum vector m (multiplied by the learning rate η), and it updates the weights by simply subtracting this momentum vector. In other words, the gradient is used as an acceleration, not as a speed.
- Due to the momentum, the optimizer may overshoot a bit, then come back, overshoot again, and oscillate like this many times before stabilizing at the minimum. This is one of the reasons why it is good to have a bit of friction in the system: it gets rid of these oscillations and thus speeds up convergence.
- The one drawback of Momentum optimization is that it adds yet another hyperparameter to tune.  

#### Nesterov Accelerated Gradient
- One small variant to Momentum optimization called Nesterov Accelerated Gradient (or Nesterov Accelerated Gradient) is almost always faster than vanilla Momentum optimization. The idea is to measure the gradient of the cost function not at the local position but slightly ahead in the direction of the momentum.  

#### AdaGrad
- Consider the elongated bowl problem again: Gradient Descent starts by quickly going down the steepest slope, then slowly goes down the bottom of the valley. It would be nice if the algorithm could detect this early on and correct its direction to point a bit more toward the global optimum.
- The AdaGrad algorithm achieves this by scaling down the gradient vector along the steepest dimensions.
- In short, this algorithm decays the learning rate, but it does so faster for steep dimensions than for dimensions with gentler slopes. This is called an adaptive learning rate. It helps point the resulting updates more directly toward the global optimum.
- One additional benefit is that it requires much less tuning of the learning rate hyperparameter η.  

#### RMSProp
- Although AdaGrad slows down a bit too fast and ends up never converging to the global optimum, the RMSProp algorithm fixes this by accumulating only the gradients from the most recent iterations (as opposed to all the gradients since the beginning of training). It does so by using exponential decay in the first step.  

#### Adam Optimization
- Adam, which stands for adaptive moment estimation, combines the ideas of Momentum optimization and RMSProp: just like Momentum optimization it keeps track of an exponentially decaying average of past gradients, and just like RMSProp it keeps track of an exponentially decaying average of past squared gradients
- **The conclusion is that you should almost always use Adam optimization.**

### Learning Rate Scheduling
- Finding a good learning rate can be tricky. If you set it way too high, training may actually diverge. If you set it too low, training will eventually converge to the optimum, but it will take a very long time. If you set it slightly too high, it will make progress very quickly at first, but it will end up dancing around the optimum, never settling down. 
- You may be able to find a fairly good learning rate by training your network several times during just a few epochs using various learning rates and comparing the learning curves. The ideal learning rate will learn quickly and converge to good solution.
- However, you can do better than a constant learning rate: if you start with a high learning rate and then reduce it once it stops making fast progress, you can reach a good solution faster than with the optimal constant learning rate. There are many different strategies to reduce the learning rate during training. These strategies are called *learning schedules*. 

### Avoiding Overfitting Through Regularization
- Deep neural networks typically have tens of thousands of parameters, sometimes even millions. With so many parameters, the network has an incredible amount of freedom and can fit a huge variety of complex datasets. But this great flexibility also means that it is prone to overfitting the training set.  

#### Early Stopping
- To avoid overfitting the training set, a great solution is early stopping: just interrupt training when its performance on the validation set starts dropping.
- Although early stopping works very well in practice, you can usually get much higher performance out of your network by combining it with other regularization techniques.  

#### ℓ1 and ℓ2 Regularization
- You can use ℓ1 and ℓ2 regularization to constrain a neural network’s connection weights (but typically not its biases).  

#### Dropout
- The most popular regularization technique for deep neural networks is arguably dropout.
- It is a fairly simple algorithm: at every training step, every neuron (including the input neurons but excluding the output neurons) has a probability *p* of being temporarily “dropped out,” meaning it will be entirely ignored during this training step, but it may be active during the next step. The hyperparameter *p* is called the dropout rate, and it is typically set to 50%. After training, neurons don’t get dropped anymore.
- The intuition behind dropout is this: Neurons trained with dropout cannot co-adapt with their neighboring neurons; they have to be as useful as possible on their own. They also cannot rely excessively on just a few input neurons; they must pay attention to each of their input neurons. They end up being less sensitive to slight changes in the inputs. In the end you get a more robust network that generalizes better.
- Another way to understand the power of dropout is to realize that a unique neural network is generated at each training step. Since each neuron can be either present or absent, there is a total of 2N possible networks (where N is the total number of droppable neurons). This is such a huge number that it is virtually impossible for the same neural network to be sampled twice. Once you have run a 10,000 training steps, you have essentially trained 10,000 different neural networks (each with just one training instance). These neural networks are obviously not independent since they share many of their weights, but they are nevertheless all different. The resulting neural network can be seen as an averaging ensemble of all these smaller neural networks.
- Dropout does tend to significantly slow down convergence, but it usually results in a much better model when tuned properly. So, it is generally well worth the extra time and effort.
- *Dropconnect* is a variant of dropout where individual connections are dropped randomly rather than whole neurons. In general dropout performs better.  

#### Max-Norm Regularization
- Another regularization technique that is quite popular for neural networks is called max-norm regularization: for each neuron, it constrains the weights *w* of the incoming connections such that ∥w∥<sub>2</sub> ≤ r, where *r* is the max-norm hyperparameter and ∥ · ∥<sub>2</sub> is the ℓ2 norm.
- Maxnorm regularization can also help alleviate the vanishing/exploding gradients problems.  

#### Data Augmentation
- One last regularization technique, data augmentation, consists of generating new training instances from existing ones, artificially boosting the size of the training set. This will reduce overfitting, making this a regularization technique. 
- The trick is to generate realistic training instances; ideally, a human should not be able to tell which instances were generated and which ones were not. Moreover, simply adding white noise will not help; the modifications you apply should be learnable (white noise is not).  


### Practical Guidelines
- The configuration in below table will work fine in most cases.
| Parameter | Default value |
| :---------------- | :--------------- |
| Initialization | He initialization |
| Activation function | ELU |
| Normalization | Batch Normalization | 
| Regularization | Dropout |
| Optimizer | Adam |
| Learning rate schedule | None |

- This default configuration may need to be tweaked:
    - If you can’t find a good learning rate (convergence was too slow, so you increased the training rate, and now convergence is fast but the network’s accuracy is suboptimal), then you can try adding a learning schedule such as exponential decay.
    - If your training set is a bit too small, you can implement data augmentation.
    - If you need a sparse model, you can add some ℓ1 regularization to the mix (and optionally zero out the tiny weights after training). If you need an even sparser model, you can try using FTRL instead of Adam optimization, along with ℓ1 regularization.
    - If you need a lightning-fast model at runtime, you may want to drop Batch Normalization, and possibly replace the ELU activation function with the leaky ReLU. Having a sparse model will also help. 

