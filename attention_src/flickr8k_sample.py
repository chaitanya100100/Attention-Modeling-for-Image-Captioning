import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from flickr8k_build_vocab import Vocabulary
from flickr8k_model import EncoderCNN, DecoderRNNWithAttention
from PIL import Image


# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

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

    # Build models
    encoder = EncoderCNN(args.encoded_image_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNNWithAttention(args.embed_size, args.attention_size, args.hidden_size, len(vocab)).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    features = encoder(image_tensor)

    sampled_seqs = decoder.sample_beam_search(features, vocab, device)
    #features = features.repeat(4,1,1,1)
    #sampled_seqs = decoder.sample(features, vocab, device)

    for s in sampled_seqs:
        # Convert word_ids to words
        sampled_caption = []
        for word_id in s:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)

        # Print out the image and the generated caption
        print (sentence)

    #image = Image.open(args.image)
    #plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='./flickr8k_models/encoder-5-124.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./flickr8k_models/decoder-5-124.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data/flickr8k_vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_size', type=int , default=224, help='input image size')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--encoded_image_size', type=int , default=14, help='dimension of encoded image')
    parser.add_argument('--attention_size', type=int , default=512, help='dimension of attention layers')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    
    args = parser.parse_args()
    main(args)
