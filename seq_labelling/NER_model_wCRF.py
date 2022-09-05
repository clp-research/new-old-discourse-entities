import torch.nn as nn
from torchcrf import CRF


class EntityLabeler(nn.Module):
    def __init__(self, vocab_size, label_size, configs):
        super(EntityLabeler, self).__init__()
        self.pre_trained = configs["pre_trained"]
        self.use_lstm = configs["lstm"]

        # embedding layer (if trained from scratch, otherwise pre-trained embeddings will be passed as input already)
        if not self.pre_trained:
            self.embedding = nn.Embedding(vocab_size, configs["embed_size"])

        # lstm or just linear layer
        if self.use_lstm:
            self.lstm = nn.LSTM(configs["embed_size"], configs["hidden"], batch_first=True)
            self.relu = nn.ReLU(configs["hidden"])
            self.linear = nn.Linear(configs["hidden"], label_size)
        else:
            self.relu = nn.ReLU(configs["embed_size"])
            self.linear = nn.Linear(configs["embed_size"], label_size)

        self.dropout = nn.Dropout(configs["dropout"])

        # CRF layer
        self.crf = CRF(label_size, batch_first=True)

    def forward(self, src_input, labels=None, masks=None):
        if self.pre_trained:
            input_representation = src_input
        else:
            input_representation = self.embedding(src_input)  # dim: batch_size x batch_max_len x embedding_dim 5 X 150 X 300

        if self.use_lstm:
            lstm_out, _ = self.lstm(input_representation)
            # apply dropout
            lstm_out = self.dropout(lstm_out)
            # lstm_token_wise = lstm_out.reshape(lstm_out.shape[0]*lstm_out.shape[1], lstm_out.shape[2])
            relu_trans = self.relu(lstm_out)
            # apply dropout
            relu_trans = self.dropout(relu_trans)
            out_linear = self.linear(relu_trans)
        else:
            relu_trans = self.relu(input_representation)
            #relu_trans = self.dropout(relu_trans)
            out_linear = self.linear(relu_trans)


        if labels is not None:
            # return negative log-likelihood (crf computes positive ll)
            return -self.crf(out_linear, labels, masks)
        else:
            # Apply the Viterbi algorithm to get the predictions. This implementation returns
            # the result as a list of lists (not a tensor), corresponding to a matrix
            # of shape (n_sentences, max_len).
            return self.crf.decode(out_linear)
