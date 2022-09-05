import argparse
import logging
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import fasttext


class EntityDataset(Dataset):
    """
    Loads pre-processed Arrau corpus and pre-trained vector representations
    Creates samples by splitting documents into context and next
    entity mention incrementally
    """
    def __init__(self, data_path, hidden_path, model, eos_vector, eos_string, device, logger, debug=False):
        self.logger = logger
        self.debug = debug
        self.device = device
        self.label = {'B-new': 1, 'B-old': 0}
        self.eos_string = eos_string
        self.eos_vector = eos_vector  # pre-extracted pre-trained representation
        self.embed_size = eos_vector.size(dim=1)
        if model == "fasttext":
            self.ft = fasttext.load_model('cc.en.300.bin')
        self.dataset, self.entity_stats = self.load_data(data_path, hidden_path, model)
        self.max_context_len = self.get_max()

    def __getitem__(self, index):
        item = self.dataset[index]
        # item is a dict containing 'context_words', 'target_words',
        # 'context_hidden', 'target_hidden' and 'label' 0(old) or 1(new)
        context = self.pad_context_vector(item['context_hidden']).to(self.device)
        target = item['target_hidden'].to(self.device)
        label = torch.FloatTensor([item['label']]).to(self.device)
        return context, target, label, index  # return index to reconstruct information for debugging

    def __len__(self):
        return len(self.dataset)

    def load_data(self, path, hidden_path, model):
        """
        Create samples by splitting input incrementally at each entity,
        storing the context representation (words for debugging/evaluation
        and hidden representations), entity representation (same way as
        context) and whether the current entity is old or new given the context
        args:
            path (str): Path to directory with one document per file
                        in CoNLL format with word in second and iob-tag
                        in last column.
            hidden_path (str): Document level extracted pre-trained
                        representations with one document per file
                        for the same documents as above.
                        (see extract_pretrained_hidden.py).
            model (str): identifier for pre-trained model
                        (if transfo-xl-wt103 or gpt2, hidden representations are used,
                        if None, random 1024d embeddings are used,
                        if fasttext, fasttext embeddings are extracted on the fly)

        return:
            samples: a list of samples, each represented by a dict containing a lists
                    of 'context_words', 'target_words', 'context_hidden' representations,
                    the averaged 'target_hidden' representation and the label 0(old) or 1(new)
                    samples[s1{ 'context_words':[w1,w2,...], 'target_words':[t1,t2,...],
                    'context_hidden':[Tensor1, Tensor2, ...] 'target_hidden':Tensor,
                    'label':0/1}]]
            stats:  a dict containing lists of the number of context entities in each sample per label

        """
        beginnings = {0: defaultdict(list), 1: defaultdict(list)}
        samples = []
        stats = {0: [], 1: []}
        content = os.listdir(path)
        for doc in content:
            file = os.path.join(path, doc)
            # skip any subdirs
            if os.path.isfile(file):
                with open(file, encoding='utf-8') as f:
                    current_sample = {"file": file}  # add file path for debugging
                    entity_counter = 0
                    # append <eos> vector to beginning of context
                    # to avoid empty context (in case sequence starts with entity)
                    context_words = [self.eos_string]
                    context_hidden = [self.eos_vector]
                    target_words = []
                    target_hidden = []
                    inside_entity = False
                    if model == "transfo-xl-wt103" or model.startswith("gpt2"):
                        # load pre-trained hidden vectors
                        try:
                            hidden_file = os.path.join(hidden_path, doc)
                            hidden_vectors = self.load_hidden_vectors(hidden_file)
                            word_index = 0  # keep track of word position in hidden vectors
                        except FileNotFoundError:
                            self.logger.error(f"Cannot find hidden file {hidden_file}\n Skipping file {file}")
                            continue
                    for line in f:
                        if line == "\n":  # newline marks end of sentence
                            # TODO check if adding sep token as sentence boundary to context makes sense
                            # context_hidden.append(self.eos_vector)
                            # context_words.append(self.eos_string)
                            continue
                        else:
                            cols = line.strip().split()
                            token = cols[1]
                            tag = cols[-1]
                            pos = cols[2]
                            baseline_tag = cols[3]
                            if model == "transfo-xl-wt103" or model.startswith("gpt2"):
                                hidden = hidden_vectors[word_index:word_index + 1, :]
                                word_index += 1
                            elif model == "fasttext":
                                hidden = self.get_ft_vector(token)
                            elif model is None:
                                hidden = torch.rand(1, 1)  # dummy vector
                            if inside_entity:
                                if tag.startswith('I'):  # complex entity
                                    # update only target entity representations
                                    target_hidden.append(hidden)
                                    target_words.append(token)
                                else:  # end of entity (either followed by another entity or outside)
                                    inside_entity = False
                                    # save current sample (copy because of incremental updates)
                                    current_sample['context_words'] = list(context_words)
                                    current_sample['context_hidden'] = torch.cat(context_hidden, dim=0)
                                    current_sample['target_words'] = list(target_words)
                                    current_sample['baseline_label'] = baseline_label
                                    current_sample['target_hidden'] = sum(target_hidden)
                                    current_sample['label'] = label
                                    current_sample['num_context_entities'] = entity_counter
                                    samples.append(current_sample)
                                    # update entity stats
                                    stats[label].append(entity_counter)

                                    current_sample = {}
                                    # append target to context for next sample
                                    context_words.extend(target_words)
                                    context_hidden.extend(target_hidden)
                                    if label:  # only increment counter for new entities
                                        entity_counter += 1
                                    # reset target for next sample
                                    target_words = []
                                    target_hidden = []
                                    # process current token
                                    if tag in ['B-new', 'B-old']:
                                        # update target representations
                                        inside_entity = True
                                        label = self.label[tag]
                                        beginnings[label][pos].append(token.lower())
                                        target_hidden.append(hidden)
                                        target_words.append(token)
                                        baseline_label = self.label[baseline_tag]
                                    else:
                                        # update context representations
                                        context_hidden.append(hidden)
                                        context_words.append(token)
                            # not inside an entity
                            # either beginning of entity or outside
                            elif tag in ['B-new', 'B-old']:
                                # update target representations
                                inside_entity = True
                                label = self.label[tag]
                                beginnings[label][pos].append(token.lower())
                                target_hidden.append(hidden)
                                target_words.append(token)
                                baseline_label = self.label[baseline_tag]
                            else:
                                # update context representations
                                context_hidden.append(hidden)
                                context_words.append(token)
                    if model == "transfo-xl-wt103" or model.startswith("gpt2"):
                        # make sure we used all hidden vectors
                        assert word_index == len(hidden_vectors)
            else:
                self.logger.debug(f"Skipping file {file}")

        if self.debug:
            for c in beginnings:
                print(f"class: {'new' if c == 1.0 else 'old'}")
                sorted_beginnings = sorted(beginnings[c], key=lambda x: len(beginnings[c][x]), reverse=True)
                for pos in sorted_beginnings:
                    print(f"pos: {pos} ({len(beginnings[c][pos])}) tokens: {set(beginnings[c][pos])}")

        return samples, stats

    def get_max(self):
        """
        Get maximum context length from dataset (for padding)
        """
        lengths = [len(x['context_words']) for x in self.dataset]
        return max(lengths)

    def plot_stats(self, mode):
        """
        Plot new/old next entity sample counts by number of context entities
        """
        new_count = self.entity_stats[1]
        old_count = self.entity_stats[0]
        max_entities = max(max(new_count, old_count))

        fig, ax = plt.subplots()
        bins = np.linspace(0, max_entities)
        plt.hist([new_count, old_count], bins, label=[f'new (total {len(new_count)})',
                                                      f'old (total {len(old_count)})'])
        ax.set_xlabel('# Context entities')
        ax.set_ylabel('# Samples')
        ax.set_title('New/old next entity samples by number of context entities')
        ax.legend()
        # plt.show()
        plt.savefig(f"{mode}_data_stats.png")
        return f"{mode}_data_stats.png"

    def pad_context_vector(self, vector):
        """
        Pad contexts to max len with zero vector.
        """
        pad_len = self.max_context_len - len(vector)
        padding = torch.zeros(pad_len, self.embed_size)
        padded_hidden = torch.cat((vector, padding), dim=0)
        return padded_hidden

    def load_hidden_vectors(self, hidden_file):
        """
        Load pre-extracted hidden representation from file
        (see extract_pretrained_hidden.py)
        """
        with open(hidden_file, "rb") as hidden_f:
            hidden_representation = pickle.load(hidden_f)
            hidden_vectors = torch.squeeze(hidden_representation, dim=0)  # remove batch dimension (was 1)
            assert self.embed_size == hidden_vectors.size(dim=1)
            hidden_vectors = hidden_vectors.detach()  # we don't want to train the hidden representations further
            return hidden_vectors

    def get_ft_vector(self, token):
        ft_vector = self.ft.get_word_vector(token)
        ft_vector = torch.from_numpy(ft_vector)
        return torch.reshape(ft_vector, (1, 300))


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Load and analyze corpus data')
    parser.add_argument('corpus', help='path to corpus files', type=str)
    args = parser.parse_args()

    # Set up logging
    logger = logging.getLogger(__name__)
    # prepare logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO  # logging.DEBUG, logging.WARN could also be set via argparser
    )
    print("Loading data...")
    test = EntityDataset(args.corpus, None, None, torch.rand(1, 1), "<eos>", None, logger, debug=True)
