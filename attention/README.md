# How to implement a Transformer network

Starting with multi headed attention

Looking at the attention is all you need paper, everything is easy to understand except for multiheaded attention. You have feedforward networks, some skip connections and normalization.

First input is key, value and query in the encoder

In the decoder has a second multi attention block that aggregates inputs from a decoder attention and output of encoders

Clarify what output embedding is

Split an embedding of size 256 and split it into 8 to have 8 layers each of size 32 and for each run an multi head attention

Take queries multiply by key and divide by root of and then multiply by value then concatenate all the split parts

Then finally run a linear layer

> We call the input the values. Some (trainable) mechanism assigns a key to each value. Then to each output, some other mechanism assigns a query. These names derive from the datastructure of a key-value store. In that case we expect only one item in our store to have a key that matches the query, which is returned when the query is executed. Attention is a softened version of this: every key in the store matches the query to some extent. All are returned, and we take a sum, weighted by the extent to which each key matches the query - Peter Bloem

## References
* [Aladdin Persson video](https://www.youtube.com/watch?v=U0s0f995w14&t=1755s)
* [Peter Bloem blog](http://peterbloem.nl/blog/transformers)
* [Query, Key and Value explained on SO](https://stats.stackexchange.com/a/424127)