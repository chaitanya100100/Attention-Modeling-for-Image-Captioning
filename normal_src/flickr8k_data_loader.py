import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from flickr8k_build_vocab import Vocabulary, Flickr8k


class Flickr8kTrainDataset(data.Dataset):

    def __init__(self, image_dir, caption_path, split_path, vocab, transform=None):
        self.image_dir = image_dir
        self.f8k = Flickr8k(caption_path=caption_path)

        with open(split_path, 'r') as f:
            self.train_imgs = f.read().splitlines()

        self.vocab = vocab
        self.transform = transform
        self.cpi = 5 # captions per image

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        fname = self.train_imgs[index//self.cpi]
        caption = self.f8k.captions[fname][index%self.cpi]

        image = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.train_imgs)*self.cpi


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = torch.tensor([len(cap) for cap in captions]).long()
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


def get_train_loader(image_dir, caption_path, train_path, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # Flickr8k caption dataset
    f8k = Flickr8kTrainDataset(image_dir=image_dir,
                       caption_path=caption_path,
                       split_path=train_path,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for Flickr8k dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    train_loader = torch.utils.data.DataLoader(dataset=f8k, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return train_loader





class Flickr8kValidationDataset(data.Dataset):

    def __init__(self, image_dir, caption_path, split_path, vocab, transform=None):
        self.image_dir = image_dir
        self.f8k = Flickr8k(caption_path=caption_path)

        with open(split_path, 'r') as f:
            self.val_imgs = f.read().splitlines()

        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        vocab = self.vocab
        fname = self.val_imgs[index]
        image = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        captions = []
        for cap in self.f8k.captions[fname]:
            # Convert caption (string) to word ids.
            tokens = nltk.tokenize.word_tokenize(str(cap).lower())
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            captions.append(caption)
            
        return image, captions

    def __len__(self):
        return len(self.val_imgs)



def get_validation_loader(image_dir, caption_path, val_path, vocab, transform, batch_size, num_workers):
    f8k = Flickr8kValidationDataset(image_dir=image_dir,
                   caption_path=caption_path,
                   split_path=val_path,
                   vocab=vocab,
                   transform=transform)
    
    def collate_fn2(data):
        images, captions = zip(*data)
        images = torch.stack(images, 0)

        return images, captions

    validation_loader = torch.utils.data.DataLoader(dataset=f8k, 
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers, 
                                          collate_fn=collate_fn2)
    return validation_loader