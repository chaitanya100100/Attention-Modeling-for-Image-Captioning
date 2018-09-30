import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from flickr8k_data_loader import get_validation_loader 
from flickr8k_build_vocab import Vocabulary
from flickr8k_model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_validation_loader(args.image_dir, args.caption_path, args.val_path, vocab, 
                             transform, args.batch_size,
                             num_workers=args.num_workers)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
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

    print(corpus_bleu(ground_truth, predicted, weights=(1, 0, 0, 0)))
    print(corpus_bleu(ground_truth, predicted, weights=(0.5, 0.5, 0, 0)))
    print(corpus_bleu(ground_truth, predicted, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0, 0)))
    print(corpus_bleu(ground_truth, predicted))
    """
    after 9 epochs:
    0.6128819934179596
    0.3944489440674209
    0.25744813154084234
    0.16841884047800518
    
    after 10 epochs:
    0.6301202185624868
    0.4098717801937141
    0.2697369272944689
    0.1770156029723617
    """

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='./flickr8k_models/encoder-10-558.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./flickr8k_models/decoder-10-558.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data/flickr8k_vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='../flickr8k/Flickr8k_Dataset', help='directory for images')
    parser.add_argument('--caption_path', type=str, default='../flickr8k/token.txt', help='path for caption file')
    parser.add_argument('--val_path', type=str, default='../flickr8k/devImages.txt', help='path for val split file')
    parser.add_argument('--image_size', type=int , default=224, help='input image size')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=384, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    print(args)
    main(args)
