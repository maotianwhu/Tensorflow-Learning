#### tensorflow.contrib这个库， 官方对它的描述是：

此目录中的任何代码未经官方支持，可能会随时更改或删除，每个目录下都有指定的所有者。
它旨在包含额外功能和贡献，最终会合并到Tensorflow中，但其接口可能仍然会发生变化，
或者需要进行一些测试，看是否可以获得更广泛的接受。所有slim依然不属于原生tensorflow。

#### 那什么是slim? slim?到底有什么用呢?

slim是一个使构建、训练、评估神经网络变得简单的库。它可以消除原生tensorflow里面很多重复的模板性的代码，
让代码更紧凑，更具备可读性。另外slim提供了很多计算机视觉方面的著名模型(VGG, AlexNet等)，
我们不仅可以直接使用，甚至能以各种方式进行扩展。

#### slim的子模块及功能介绍:

arg_scope: provides a new scope named arg_scope that allows a user to define default 
arguments for specific operations within that scope, 它用来控制每一层的默认超参数的。

data: contains TF-slim's dataset definition, data providers, parallel_reader, and decoding utilities.

evaluation: contains routines for evaluating models.

**layers:** contains high level layers for building models using tensorflow.

learning: contains routines for training models.

losses: contains commonly used loss functions.

metrics: contains popular evaluation metrics.

**nets:** contains popular network definitions such as VGG and AlexNet models.

**queues:** provides a context manager for easily and safely starting and closing QueueRunners.
 文本管理队列，比较有用.
 
regularizers: contains weight regularizers.

**variables:** provides convenience wrappers for variable creation and manipulation.



