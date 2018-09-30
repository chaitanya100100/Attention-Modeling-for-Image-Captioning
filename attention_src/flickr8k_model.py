import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# In original implementation, length wise sorting of batch is done in forward function
# I have done that in dataloader itself
# Original implementation use dropout layer in decoder but I am not using it

# repeat 'tensor' 'times' times along 'dim' dimension
def _inflate(tensor, times, dim):
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size):
        """Load the pretrained ResNet-101 and remove last layers."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        #modules = list(resnet.children())[:-2]      # output of size (image_size/32)*(image_size/32)*2048
        modules = list(resnet.children())[:-3]      # output of size (image_size/16)*(image_size/16)*1024
        self.resnet = nn.Sequential(*modules)
        
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # optional : can set requires_grad=True for some layers to finetune
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            feat_vecs = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        feat_vecs = self.adaptive_pool(feat_vecs)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        feat_vecs = feat_vecs.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return feat_vecs

    
class Attention(nn.Module):
    def __init__(self, encoder_size, hidden_size, attention_size):
        """Set the layers"""
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_size, attention_size) # linear layer to transform encoded image
        self.hidden_att = nn.Linear(hidden_size, attention_size) # linear layer to transform previous hidden output
        self.full_att = nn.Linear(attention_size, 1) # linear layer to calculate pre-softmax values
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) # because dim0 is for batch 

    def forward(self, encoder_out, hidden_out):
        """Generate attention encoded input from encoder output and previous hidden output"""
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_size)
        att2 = self.hidden_att(hidden_out)  # (batch_size, attention_size)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_size)
        return attention_weighted_encoding, alpha


class DecoderRNNWithAttention(nn.Module):
    def __init__(self, embed_size, attention_size, hidden_size, vocab_size, encoder_size=1024, max_seg_length=40):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNNWithAttention, self).__init__()
        
        self.attention = Attention(encoder_size=encoder_size, hidden_size=hidden_size, attention_size=attention_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstmcell = nn.LSTMCell(embed_size+encoder_size, hidden_size, bias=True)
        
        self.init_hidden = nn.Linear(encoder_size, hidden_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_cell = nn.Linear(encoder_size, hidden_size)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(hidden_size, encoder_size)  # linear layer to create a sigmoid-activated gate according to paper
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()
        
        self.vocab_size = vocab_size
        self.max_seg_length = max_seg_length

    def init_weights(self):
        """Initialize the weights of learnable layers"""
        self.embed.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1,0.1)

    def init_hidden_state(self, encoder_out):
        """Mean of encoder output features as initial hidden and cell state"""
        mean_encoder_out = encoder_out.mean(dim=1)
        hidden = self.init_hidden(mean_encoder_out)
        cell = self.init_cell(mean_encoder_out)
        return hidden, cell

    def forward(self, encoder_out, captions, lengths, device):
        """Decode image feature vectors and generates captions."""
        batch_size, encoder_size, vocab_size = encoder_out.size(0), encoder_out.size(-1), self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_size)
        num_pixels = encoder_out.size(1)

        # dataloader has sorted the batch according to caption lengths, hence no need to sort here
        #lengths, sort_ind = lengths.squeeze(1).sort(dim=0, descending=True)
        #encoder_out = encoder_out[sort_ind]
        #captions = captions[sort_ind]
        
        embeddings = self.embed(captions) # (batch_size, max_caption_length, embed_size)
        hidden, cell = self.init_hidden_state(encoder_out) # (batch_size, hidden_size)
        
        lengths = [l - 1 for l in lengths]
        max_length = max(lengths)
        
        predictions = torch.zeros(batch_size, max_length, vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_length, num_pixels).to(device)
        
        for t in range(max_length):
            batch_size_t = sum([l > t for l in lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], hidden[:batch_size_t])
            gate = self.sigmoid(self.f_beta(hidden[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_size)
            attention_weighted_encoding = gate * attention_weighted_encoding
            hidden, cell = self.lstmcell(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (hidden[:batch_size_t], cell[:batch_size_t]))  # (batch_size_t, hidden_size)
            predictions[:batch_size_t, t, :] = self.fc(hidden)  # (batch_size_t, vocab_size)
            alphas[:batch_size_t, t, :] = alpha

        return predictions, captions, lengths, alphas
    

    def sample(self, encoder_out, vocab, device):
        """Generate captions for given image features using greedy search."""
        batch_size = encoder_out.size(0)
        encoder_size = encoder_out.size(-1)
        
        encoder_out = encoder_out.view(batch_size, -1, encoder_size)
        
        hidden, cell = self.init_hidden_state(encoder_out) # (batch_size, hidden_size)
        inputs = self.embed(torch.tensor([vocab('<start>')]).to(device)).repeat(batch_size, 1)
        
        sampled_ids = []
        for t in range(self.max_seg_length):
            attention_weighted_encoding, alpha = self.attention(encoder_out, hidden)
            gate = self.sigmoid(self.f_beta(hidden))
            attention_weighted_encoding = gate * attention_weighted_encoding
            hidden, cell = self.lstmcell(
                torch.cat([inputs, attention_weighted_encoding], dim=1),
                (hidden, cell))

            _, predicted = self.fc(hidden).max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)

        sampled_ids = torch.stack(sampled_ids, 1).tolist() # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = [[vocab('<start>')]+s for s in sampled_ids]
        return sampled_ids

    
    def sample_beam_search(self, encoder_out, vocab, device, beam_size=4):
        k = beam_size
        vocab_size = len(vocab)
        encoder_size = encoder_out.size(-1)
        encoder_out = encoder_out.view(1, -1, encoder_size)
        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(k, num_pixels, encoder_size)  # (k, num_pixels, encoder_dim)
        k_prev_words = torch.LongTensor([[vocab('<start>')]] * k).to(device)  # (k, 1)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
        complete_seqs = list()
        complete_seqs_scores = list()
        
        hidden, cell = self.init_hidden_state(encoder_out)
        step = 1
        while True:
            embeddings = self.embed(k_prev_words).squeeze(1)
            awe, _ = self.attention(encoder_out, hidden)
            gate = self.sigmoid(self.f_beta(hidden))
            awe = gate * awe
            hidden, cell = self.lstmcell(torch.cat([embeddings, awe], dim=1), (hidden, cell))

            scores = self.fc(hidden)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0)  # (s)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != vocab('<end>')]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            hidden = hidden[prev_word_inds[incomplete_inds]]
            cell = cell[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > self.max_seg_length:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        return [seq]
    
        return complete_seqs

    """
    def sample_batch_beam_search(self, encoder_out, vocab, device, beam_size=4):
        print("Incomplete")
        return []
        k = beam_size
        batch_size = encoder_out.size(0)
        encoder_size = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_size)
        num_pixels = encoder_out.size(1)

        encoder_out = _inflate(encoder_out, k, dim=0)
        hidden, cell = self.init_hidden_state(encoder_out)

        selected_words = torch.LongTensor(k*batch_size, self.max_seg_length).fill_(0)
        selected_words[:, 0] = vocab.word2idx['<start>']
        
        predecessors = torch.LongTensor(k*batch_size, self.max_seg_length).fill_(-1)
        
        sequence_scores = torch.zeros(batch_size * k, 1)
        pos_index = (torch.LongTensor(range(batch_size)) * k).view(-1, 1)
        
        for t in range(self.max_seg_length-1):
            attention_weighted_encoding, alpha = self.attention(encoder_out, hidden)
            gate = self.sigmoid(self.f_beta(hidden))
            attention_weighted_encoding = gate * attention_weighted_encoding
            inputs = selected_words[:, t].clone().to(device)
            inputs = self.embed(inputs).squeeze(1)
            hidden, cell = self.lstmcell(
                torch.cat([inputs, attention_weighted_encoding], dim=1),
                (hidden, cell))
            out_scores = F.log_softmax(self.fc(hidden), dim=1).cpu()
            V = out_scores.size(-1)

            # To get the full sequence scores for the new candidates, add the local scores to the predecessor scores 
            sequence_scores = _inflate(sequence_scores, V, 1)
            sequence_scores += out_scores
            top_scores, candidates = sequence_scores.view(batch_size, -1).topk(k, dim=1)
            print(top_scores)
            selected_words[:, t+1] = (candidates % V).view(batch_size * k)
            sequence_scores = top_scores.view(batch_size * k, 1)
            
            predecessors[:, t+1] = (candidates / V + pos_index.expand_as(candidates)).view(batch_size * k)
            hidden = hidden[predecessors[:, t+1].squeeze()]

        
        return []
    """