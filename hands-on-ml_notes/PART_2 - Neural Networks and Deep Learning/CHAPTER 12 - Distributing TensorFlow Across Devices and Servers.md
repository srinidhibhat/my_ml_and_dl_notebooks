## CHAPTER 12 - Distributing TensorFlow Across Devices and Servers

- TensorFlow’s support of distributed computing is one of its main highlights compared to other neural network frameworks. It gives you full control over how to split (or replicate) your computation graph across devices and servers, and it lets you parallelize and synchronize operations in flexible ways so you can choose between all sorts of parallelization approaches.

### Multiple Devices on a Single Machine
- You can often get a major performance boost simply by adding GPU cards to a single machine. In fact, in many cases this will suffice; you won’t need to use multiple machines at all.
- If you don’t own any GPU cards, you can use a hosting service with GPU capability such as Amazon AWS.

### Parallel Execution
- When TensorFlow runs a graph, it starts by finding out the list of nodes that need to be evaluated, and it counts how many dependencies each of them has. TensorFlow then starts evaluating the nodes with zero dependencies (i.e., source nodes). If these nodes are placed on separate devices, they obviously get evaluated in parallel. If they are placed on the same device, they get evaluated in different threads, so they may run in parallel too (in separate GPU threads or CPU cores).

### Multiple Devices Across Multiple Servers
- To run a graph across multiple servers, you first need to define a cluster. A cluster is composed of one or more TensorFlow servers, called tasks, typically spread across several machines. Each task belongs to a job. A job is just a named group of tasks that typically have a common role, such as keeping track of the model parameters (such a job is usually named "ps" for parameter server), or performing computations (such a job is usually named "worker").
