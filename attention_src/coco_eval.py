import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from coco_data_loader import get_validation_loader 
from coco_build_vocab import Vocabulary
from coco_model import EncoderCNN, DecoderRNNWithAttention
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # Build data loader
    data_loader = get_validation_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             num_workers=args.num_workers)
    
    # Build models
    encoder = EncoderCNN(args.encoded_image_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNNWithAttention(args.embed_size, args.attention_size, args.hidden_size, len(vocab)).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    
    ground_truth = []
    predicted = []
    for i, (images, captions) in enumerate(data_loader):
        # Set mini-batch dataset
        images = images.to(device)
        features = encoder(images)
        sampled_seq = decoder.sample_beam_search(features, vocab, device)
        
        sampled_seq = sampled_seq[0][1:-1]
        captions = [c[1:-1] for c in captions[0]]
        
        ground_truth.append(captions)
        predicted.append(sampled_seq)


    print(i)        
    print(corpus_bleu(ground_truth, predicted, weights=(1, 0, 0, 0)))
    print(corpus_bleu(ground_truth, predicted, weights=(0.5, 0.5, 0, 0)))
    print(corpus_bleu(ground_truth, predicted, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0, 0)))
    print(corpus_bleu(ground_truth, predicted))
    """
    trained on crop-----
    on train dataset:
    0.7312313463196163
    0.54945515233551
    0.41614422602788537
    0.3204806090120215

    after 9 epochs: 
    0.7099924560680448
    0.5185967595065426
    0.3822381823574307
    0.2869338905515127
    
    after 7 epochs:
    0.7218095584484059
    0.5300911984502783
    0.3921688404165646
    0.2946089918301266

    

    trained on resized-----
    after 10 epochs:
    0.7191875542290782
    0.5274523016789623
    0.38932065006088895
    0.29216110168510645

    after 7 epochs:
    0.7176259868196907
    0.5261198363362601
    0.3896601607227492
    0.2926295563605041
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='./coco_models_resized/encoder-7-3174.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./coco_models_resized/decoder-7-3174.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data/coco_vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='../mscoco/resized_val2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='../mscoco/annotations/captions_val2014.json', help='path for train annotation json file')
    parser.add_argument('--image_size', type=int , default=224, help='input image size')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--encoded_image_size', type=int , default=14, help='dimension of encoded image')
    parser.add_argument('--attention_size', type=int , default=384, help='dimension of attention layers')
    parser.add_argument('--hidden_size', type=int , default=384, help='dimension of lstm hidden states')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    print(args)
    main(args)
