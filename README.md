# Attention-Modeling
Pytorch implementation of Attention Modeling for Image Captioning


### Implementation
- A normal CNN-RNN architecture for image captioning
- Implemenation of Visual Attention for image captioning described in [Show, Attend and Tell](https://arxiv.org/abs/1502.03044)

### Results

I have validated the methods on standard Flickr8k and MSCOCO dataset. They achieves state of the art accuracy. Results are as follows:

Method | Normal CNN-RNN Architecture | Attention Mechanism |
----- | --------------------------- | ------------------- |
Score |   MSCOCO  |    Flickr8k     | MSCOCO  |  Flickr8k |
----- | --------------------------- | ------------------- |
BLEU-1|   0.705   |     0.630       | 0.731   |   0.655   |
BLEU-4|   0.265   |     0.177       |  0.320  |   0.218   |


###### Normal CNN-RNN architecture

- For MSCOCO dataset
  - BLEU-1 : 0.705
  - BLEU-4 : 0.265

- For Flickr8k dataset
  - BLEU-1 : 0.630
  - BLEU-4 : 0.177

###### Visual Attention architecture

- For MSCOCO dataset
  - BLEU-1 : 0.731
  - BLEU-4 : 0.320

- For Flickr8k dataset
  - BLEU-1 : 0.655
  - BLEU-4 : 0.218

### TO DO
- Add attention visualization utility 
