# Fast ai important highlights

## Intro
https://course.fast.ai/videos/?lesson=1

GPUs are fast matrix multiplication machines

* ML not so great when decisions are already fast or easy to make
* Universal approximation theorem 
* Universal input and output functions, showcase 
* Reshape images of certain size in forward function to make them fit
* Picture of neural network that works with several data structures

## Lesson 3
* How to get more data: data augmentation, build a popular product, self supervised learning
* Broadcasting
* Loss vs activation function - make a tour of them
* Learning rate

## Lesson 4
* View in pytorch
* Data loader in pytorch
* Why batch sizes? purely for GPUs

## Lesson 6
* Learning rate schedulers - 1 cycle, exponential decay etc
* fp16
* fine tuning (gradual unfreezing) vs training
* dataset vs data loader
* regression vs classification loss functions
* Embedding matrix (key value store)

## Lesson 7
* tokenization
* perplexity
* recurrent network
* aggregating inputs `torch.stack` or `torch.cat` or `+`
* transformer and go over einsum notation for aggregations
