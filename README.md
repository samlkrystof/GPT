## Building GPT from scratch

---

This repository contains GPT style Transformer with only Decoder module. The base of this model is from Andrej Karpathy's [video on GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY). I have made some modifications to make it better.

Right now I'm working on different tokenization methods, which will be added soon. For now, the model is trained on character level. 

<b>The tokenizers are only for educational purposes, training large text corpuses is not possible with them.</b>

File [shakespeare.txt](https://github.com/samlkrystof/GPT/blob/master/shakespeare.txt) contains all texts written by William Shakespeare, these texts were used to train the model.

File [output.txt](https://github.com/samlkrystof/GPT/blob/master/output.txt) contains outputs of the model after training with configuration provided in the file. Model was trained about an hour on Tesla T4 GPU provided in Google Colab.

Bigram contains simplest possible model to generate text.

