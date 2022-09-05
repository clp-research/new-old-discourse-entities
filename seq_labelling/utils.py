
import torch
import os
import pickle
import fasttext
import fasttext.util
from torch.utils.data import Dataset
from transformers import TransfoXLModel, TransfoXLTokenizer, GPT2Tokenizer, GPT2Model


class Vocabulary(object):
    def __init__(self):
        self.token_to_id = {'[PAD]': 0, '[UNK]': 1}
        self.id_to_token = {}

    def generate_ids_tokens(self, data_dict):
        for doc in data_dict:
            data_dict[doc]["ids"] = []
            for sent in data_dict[doc]['seqs']:
                for word in sent:
                    # add to word-id dict
                    if word.lower() not in self.token_to_id:
                        self.token_to_id[word.lower()] = len(self.token_to_id)
        # create id2word dict
        self.id_to_token = {id: tok for tok, id in self.token_to_id.items()}

    def create_lookup(self, data_dict):
        for doc in data_dict:
            data_dict[doc]["lookups"] = []
            for sent in data_dict[doc]["seqs"]:
                ids = self.token2ids(sent)
                # store it in data_dict
                data_dict[doc]["lookups"].append(ids)

    def token2ids(self, token_list):
        return [self.token_to_id[word.lower()] if word.lower() in self.token_to_id else self.token_to_id["[UNK]"] for word in token_list]


class EntityDataset(Dataset):
    def __init__(self, mode, config, device, logger, vocab, padding):
        self.mode = mode
        self.pre_trained = config["pre_trained"]
        self.contextualized = config["contextualized"]
        self.embed_size = config["embed_size"]
        self.gold = config["gold_column"]
        self.transformer = config["transformer"]
        self.baseline = True if "baseline" in config else False
        self.device = device
        self.padding = padding
        self.data_dict = {} # store data document wise
        self.data = [] # store data sentence wise

        if config["gold_column"] == -1:
            self.label_to_id = {'O': 0, 'B-new': 1, 'B-old': 2, 'I-new': 3, 'I-old': 4}
            self.id_to_label = {0: 'O', 1: 'B-new', 2: 'B-old', 3: 'I-new', 4: 'I-old'}
        if config["gold_column"] == -3:
            self.label_to_id = {'O': 0, 'B': 1, 'I': 2}
            self.id_to_label = {0: 'O', 1: 'B', 2: 'I'}

        # maximum sequence length
        self.max_token = 0

        logger.info(f'Loading {self.mode} dataset...')
        path = config["data"] + self.mode

        # load data
        self.load(path)
        # generate ids for labels
        # self.generate_ids_labels(self.data_dict)
        self.label_size = len(self.label_to_id)

        if self.pre_trained:
            self.pad_vector = torch.zeros(1, self.embed_size)
            if self.contextualized:
                logger.info(f'Extracting embeddings from pre-trained model...')
                # extract hidden vectors from file (prepared by extract_pretrained_hidden.py)
                self.hidden_vectors_path = config["hidden_vectors"] + self.mode
                self.extract_hidden()  # extends self.data_dict
            else:
                logger.info(f'Using fixed embeddings...')
                self.fixed_vectors_model = fasttext.load_model(config["fixed_vectors"] + config["ft_model"])
                self.extract_fixed_embedds()  # extends self.data_dict
            self.data = self.flatten_data()

        # not pre-trained
        else:
            if mode == "train":
                vocab.generate_ids_tokens(self.data_dict)
                self.vocab_size = len(vocab.token_to_id)
            vocab.create_lookup(self.data_dict)
            self.pad_id = vocab.token_to_id['[PAD]']
            self.data = self.flatten_data()

        logger.info('Finished!')


    def load(self, path):
        """
        Load data document wise
        path: the path to a folder with document wise data (conll style files)
        fill data_dict as follows: dict{ doc: {'seqs': [[],[],...],'tags': [[],[],...]}}
        """
        for file in os.listdir(path):
            example_seqs = []
            example_tags = []
            if self.baseline:
                example_baseline_tags = []
            current_doc = {}

            with open(path + "/" + file, encoding='utf-8') as f:
                current_sequence = []
                current_tags = []
                if self.baseline:
                    current_baseline_tags = []
                for line in f:
                    if line.strip() == "":
                        example_seqs.append(current_sequence)
                        example_tags.append(current_tags)
                        if len(current_sequence) > self.max_token:
                            self.max_token = len(current_sequence)
                        current_sequence = []
                        current_tags = []
                        if self.baseline:
                            example_baseline_tags.append(current_baseline_tags)
                            current_baseline_tags = []
                    else:
                        cols = line.strip().split("\t")
                        token = cols[1]
                        tag = cols[self.gold]
                        baseline_tag = cols[3]
                        current_sequence.append(token)
                        current_tags.append(tag)
                        if self.baseline:
                            current_baseline_tags.append(baseline_tag)

                current_doc['seqs'] = example_seqs
                current_doc['tags'] = example_tags
                if self.baseline:
                    current_doc['baseline_tags'] = example_baseline_tags
                self.data_dict[file] = current_doc

    def extract_fixed_embedds(self):
        for doc in self.data_dict:
            self.data_dict[doc]["hidden"] = []
            for sent in self.data_dict[doc]["seqs"]:
                embedds = []
                for token in sent:
                    if token in self.fixed_vectors_model:
                        np_vector = self.fixed_vectors_model[token]
                        embedds.append(np_vector)
                    else:
                        np_vector = self.fixed_vectors_model["[UNK]"]
                        embedds.append(np_vector)
                # store it in self.data_dict
                tens = torch.as_tensor(embedds)
                self.data_dict[doc]["hidden"].append(tens)


    def extract_hidden(self):
        """
        Add pre-trained hidden representation to data_dict
        as follows: dict{ doc: {'seqs': [[1,2,3],[4,5,6],...],'tags': [[],[],...], 'hidden': [[],[],...]}}
        """
        # extract document-wise hidden states
        for doc in self.data_dict:
            word_doc = []
            for sent in self.data_dict[doc]["seqs"]:
                for word in sent:
                    word_doc.append(word)
            pickle_vectors = open(self.hidden_vectors_path + "/" + doc, "rb")
            hidden_vectors = pickle.load(pickle_vectors)
            hidden_vectors = torch.squeeze(hidden_vectors, dim=0)  # remove batch dimension (was 1)
            hidden_vectors = hidden_vectors.detach()

            # store hidden sequence, split into sentence list structure again
            self.data_dict[doc]['hidden'] = []
            start_i = 0
            for seq in self.data_dict[doc]['seqs']:
                end_i = start_i + len(seq)
                hidden_sequence = hidden_vectors[start_i:end_i, :]
                self.data_dict[doc]['hidden'].append(hidden_sequence)
                start_i += len(seq)


    def flatten_data(self):

        sequences = [s for doc in self.data_dict for s in self.data_dict[doc]['seqs']]
        tags = [t for doc in self.data_dict for t in self.data_dict[doc]['tags']]

        if self.baseline:
            baseline = [b for doc in self.data_dict for b in self.data_dict[doc]['baseline_tags']]
            return list(zip(sequences, tags, baseline))

        if self.pre_trained:
            hidden = [h for doc in self.data_dict for h in self.data_dict[doc]['hidden']]
            return list(zip(sequences, tags, hidden))
        else:
            word_ids = [l for doc in self.data_dict for l in self.data_dict[doc]['lookups']]
            return list(zip(sequences, tags, word_ids))

    def generate_ids_labels(self):
        """
        ({doc_id: {'seqs': [[],[]...], 'tags': [[],[]...]}})) -> update global dicts
        """

        for doc in self.data_dict:
            labels = self.data_dict[doc]['tags']
            for tags_list in labels:
                for tag in tags_list:
                    if tag not in self.label_to_id:
                        tag_id = len(self.label_to_id)
                        self.label_to_id[tag] = tag_id
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}


    def ids2token(self, id_list):
        if self.pre_trained:
            return self.tokenizer.convert_ids_to_tokens(id_list)
        else:
            return [self.id_to_token.get(id) for id in id_list.tolist()]

    def label2ids(self, label_list):
        return [self.label_to_id[tag] for tag in label_list]

    def pad_hidden(self, hidden):

        sequence = torch.as_tensor(hidden)

        masks = torch.ones(self.max_token, dtype=torch.uint8) # init masks

        # truncate longer sequences
        if len(sequence) >= self.max_token:
            return sequence[:self.max_token, :], masks # no masks

        if self.padding:
            num_extra_padding = self.max_token - len(sequence)
            extra_padding = torch.cat(num_extra_padding * [self.pad_vector], dim=0)
            padded_sequence = torch.cat((sequence, extra_padding), dim=0)
            # update masks
            masks[len(sequence):self.max_token] = 0
            return padded_sequence, masks
        else:
            masks = torch.ones(len(sequence), dtype=torch.uint8)  # init masks
            return sequence, masks

    def pad(self, sequence, pad_token):

        masks = torch.ones(self.max_token, dtype=torch.uint8)  # init masks
        # truncate longer sequences
        if len(sequence) > self.max_token:
            return sequence[:self.max_token], masks
        # pad shorter sequences
        if self.padding:
            # update masks
            masks[len(sequence):self.max_token] = 0
            padded_sequence = sequence + [pad_token] * (self.max_token - len(sequence))
            return padded_sequence, masks
        else:
            masks = torch.ones(len(sequence), dtype=torch.uint8)  # init masks
            return sequence, masks

    def preprocess_sentence(self, item):

        if self.pre_trained:
            tokens, tags, hidden = item
            padded_input, masks = self.pad_hidden(hidden)
            token_tensor = padded_input.to(self.device)
        else:
            tokens, tags, token_ids = item
            padded_input, masks = self.pad(token_ids, self.pad_id)
            token_tensor = torch.tensor(padded_input, dtype=torch.long).to(self.device)

        # todo: here the masks are not needed for tokens and labels: pad_hidden vs pad could be simplified/merged
        label_ids = self.label2ids(tags)
        padded_label_ids, masks_labels = self.pad(label_ids, 0)
        label_tensor = torch.as_tensor(padded_label_ids, dtype=torch.long).to(self.device)
        padded_tokens, masks_tokens = self.pad(tokens, "[PAD]")

        assert len(token_tensor) == len(label_tensor) == len(masks) == len(padded_tokens)
        return token_tensor, label_tensor, masks.to(self.device), padded_tokens

    def __getitem__(self, idx):
        if self.baseline:
            return self.data[idx]
        processed_hidden, processed_labels, processed_masks, tokens = self.preprocess_sentence(self.data[idx])
        return processed_hidden, processed_labels, processed_masks, tokens

    def __len__(self):
        return len(self.data)


