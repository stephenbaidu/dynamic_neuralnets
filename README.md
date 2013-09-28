## Dynamic Neural Networks
##### A simple re-configurable neural networks that uses backpropagation. This implementation saves you from having to write code for every structure that you want to try out.

</br>

Based on neural networks exercise on [Coursera](https://www.coursera.org/course/ml)

</br>

Allows you to re-configure the structure of your neural networks.
[400, 3, 3, 10]: means 400 nodes for the input layer, 10 nodes for the output layer and 3 nodes for both hidden layers.

Hence, you can test other configurations like:
[400, 5, 5, 10]
[400, 5, 20, 5, 10]
[400, 2, 10]
etc.

### Running the scripts

From terminal, execute:

    $ git clone ggit://github.com/stephenbaidu/dynamic_neuralnets.git
    $ cd dynamic_neuralnets
    $ octave nnStart.m