import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from flickr8k_data_loader import get_train_loader 
from flickr8k_build_vocab import Vocabulary
from flickr8k_model import EncoderCNN, DecoderRNNWithAttention
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_train_loader(args.image_dir, args.caption_path, args.train_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models
    encoder = EncoderCNN(args.encoded_image_size).to(device)
    decoder = DecoderRNNWithAttention(args.embed_size, args.attention_size, args.hidden_size, len(vocab)).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    params = list(decoder.parameters()) + list(encoder.adaptive_pool.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)

            # Forward, backward and optimize
            features = encoder(images)
            scores, captions, lengths, alphas = decoder(features, captions, lengths, device)
            
            targets = captions[:, 1:]
            # Remove padded words to calculate score
            targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]
            scores = pack_padded_sequence(scores, lengths, batch_first=True)[0]

            # cross entropy loss and doubly stochastic attention regularization
            loss = criterion(scores, targets)
            loss += 1.0 * ((1 - alphas.sum(dim=1))**2).mean()
            
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if (i+1) % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i+1, total_step, loss.item(), np.exp(loss.item())))

                
            # Save the model checkpoints
            if (i+1 + epoch*total_step) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./flickr8k_models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='./data/flickr8k_vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='../flickr8k/Flickr8k_Dataset', help='directory for images')
    parser.add_argument('--caption_path', type=str, default='../flickr8k/token.txt', help='path for caption file')
    parser.add_argument('--train_path', type=str, default='../flickr8k/trainImages.txt', help='path for train split file')
    parser.add_argument('--image_size', type=int , default=224, help='input image size')
    parser.add_argument('--log_step', type=int , default=20, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--encoded_image_size', type=int , default=14, help='dimension of encoded image')
    parser.add_argument('--attention_size', type=int , default=384, help='dimension of attention layers')
    parser.add_argument('--hidden_size', type=int , default=384, help='dimension of lstm hidden states')
    
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    args = parser.parse_args()
    print(args)
    main(args)
