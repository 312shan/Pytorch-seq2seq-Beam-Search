# Pytorch-seq2seq-Beam-Search
Seq2Seq model with attention and Greedy Search / Beam Search for neural machine translation in PyTorch.

This implementation focuses on the following features:

- Modular structure to be used in other projects
- Minimal code for readability
- Full utilization of batches and GPU.
- Decoding Method Greedy Search
- Decoding Method Beam Search

This implementation relies on [torchtext](https://github.com/pytorch/text) to minimize dataset management and preprocessing parts.

## Seq2Seq Model description
The main structure of seq2seq is adopt in [seq2seq](https://github.com/keon/seq2seq)
* Encoder: Bidirectional GRU
* Decoder: GRU with Attention Mechanism
* Attention

![](http://www.wildml.com/wp-content/uploads/2015/12/Screen-Shot-2015-12-30-at-1.16.08-PM.png)

## Decoding Method
* Greedy Search
* Beam Search

## Requirements

* CUDA
* Python 3.6
* PyTorch 1.4
* torchtext
* Spacy
* numpy
* Visdom (optional)

download tokenizers by doing so:
```
python -m spacy download de
python -m spacy download en
```

## TODO
1. add logging
2. change to support gpu and cpu, currently is gpu based code implementation
3. Do Length normalization experiment on Beam-Search And Compare

## References
Based on the following implementations
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* [seq2seq](https://github.com/keon/seq2seq)
* [PyTorch-Beam-Search-Decoding](https://github.com/budzianowski/PyTorch-Beam-Search-Decoding)
* [Conditional Generation by RNN & Attention](https://www.youtube.com/watch?v=f1KUUz7v8g4)
* [cs224n-2019-notes06-NMT_seq2seq_attention](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf)

