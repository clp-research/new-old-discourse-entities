# -*- encoding: utf-8 -*-
"""
Date: February 23, 2022

Reads the ARRAU data, tokenizes it, and extracts the corresponding hidden representations.
Tokenized and hidden_representations are dumped in a new file per doc each.
Labels are remapped and dumped in new files per doc too.

example arguments:
--data_in
data/arrau_processed/heads
--model
gpt2
--data_out_hidden
data/hidden_states/arrau_processed/heads
--data_out_tokens
data/tokenized/arrau_processed/heads
"""

import os
import pickle
import argparse
import torch
import re
from transformers import TransfoXLTokenizer, TransfoXLModel, GPT2Tokenizer, GPT2Model


def load(path):
    data_by_doc = {}
    data_by_sentence = {}

    files = os.listdir(path)
    for file in files:
        doc_no_sents = []
        current_doc_tokens = []
        sent_tokens = []
        with open(path + "/" + file, encoding='utf-8') as f:
            for line in f:
                if line == "\n":
                    current_doc_tokens.append(sent_tokens)
                    sent_tokens = []
                else:
                    cols = line.strip().split("\t")
                    token = cols[1]
                    pos = cols[2]
                    pos_bas = cols[3]
                    iob = cols[4]
                    coref = cols[5]
                    label = cols[-1]
                    sent_tokens.append((token, pos, pos_bas, iob, coref, label))
                    doc_no_sents.append(token)

            data_by_sentence[file] = current_doc_tokens
            data_by_doc[file] = doc_no_sents

    return data_by_doc, data_by_sentence


def add_pair(i, j, d):
    if i in d:
        d[i].append(j)
    else:
        d[i] = [j]


def postprocess_tokenized(original, tokenized, model):
    #TODO: clean_tok seems to be an unused copy of tokenized
    clean_orig = []
    clean_tok = []

    for z in range(len(original)):
        original[z] = original[z].replace("``", '"').replace("`", "'").replace("''", '"')
        clean_orig.append(original[z])

    for z in range(len(tokenized)):
        if model == "transfo-xl-wt103":
            tokenized[z] = tokenized[z].replace("@", "", 2)
            clean_tok.append(tokenized[z])

    return clean_orig, tokenized


def count(words):
    counts = {}
    for w in words:
        if w in counts:
            counts[w] += 1
        else:
            counts[w] = 1
    return counts


def find_anchors(original, tokenized):
    counts_original = count(original)
    counts_tokenized = count(tokenized)

    anchor_words = []

    for w in counts_original:
        if w in counts_tokenized:
            if counts_original[w] == counts_tokenized[w]:
                anchor_words.append(w)

    # add indexes of anchor points
    idx_original = []
    idx_tokenized = []

    for i in range(len(original)):
        if original[i] in anchor_words:
            idx_original.append(i)
    for i in range(len(tokenized)):
        if tokenized[i] in anchor_words:
            idx_tokenized.append(i)

    assert len(idx_original) == len(idx_tokenized)

    return idx_original, idx_tokenized


def produce_mapping_dic(doc_list):
    mapping = {}
    for i_tok in range(len(doc_list)):
        i_unt = doc_list[i_tok]
        if i_unt in mapping:
            mapping[i_unt].append(i_tok)
        else:
            mapping[i_unt] = [i_tok]
    return mapping


def produce_all_partitions(array, len_original):
    n = len(array)
    partitions = []

    for partition_index in range(2 ** (n-1)):
        # current partition, e.g., [['a', 'b'], ['c', 'd', 'e']]
        partition = []
        # used to accumulate the subsets, e.g., ['a', 'b']
        subset = []
        for position in range(n):

            subset.append(array[position])

            # check whether to "break off" a new subset
            if 1 << position & partition_index or position == n-1:
                partition.append(subset)
                subset = []

        if len(partition) == len_original:
            partitions.append(partition)

    return partitions


def map_versions(original, tokenized, model_name):
    index_list = [-1] * len(tokenized)

    if model_name.startswith("gpt2"):
        word_id = 0
        for token_id, token in enumerate(tokenized):
            if token_id == 0:
                # the first token is not marked with the "Ġ" symbol
                index_list[token_id] = word_id
                continue
            # the tokenizer adds "Ġ" to the first token of every word (but not to following sub-word tokens)
            if token.startswith("Ġ"):
                # we are at the beginning of a new word
                word_id += 1
            index_list[token_id] = word_id

    elif model_name == "transfo-xl-wt103":

        anchors_original, anchors_tokenized = find_anchors(original, tokenized)

        for i, j in zip(anchors_original, anchors_tokenized):
            index_list[j] = i

        #zone = start_original_i, end_original_i, start_tokenized_i, end_tokenized_i
        w = 0
        while w < len(index_list):
            if index_list[w] >= 0:
                w += 1
                continue
            start_tokenized_i = w
            if w == 0:
                start_original_i = 0
            else:
                start_original_i = index_list[w-1] + 1
            z = start_tokenized_i+1

            while z < len(index_list) and index_list[z] == -1:
                z += 1
            end_tokenized_i = z

            if z < len(index_list):
                end_original_i = index_list[z]
            else:
                end_original_i = len(original)

            # unk and 1-to-many
            if end_original_i - start_original_i == 1:
                for idx in range(start_tokenized_i, end_tokenized_i):
                    index_list[idx] = start_original_i

            elif end_original_i - start_original_i == end_tokenized_i - start_tokenized_i:
                idx_original = start_original_i
                for idx_tokenized in range(start_tokenized_i, end_tokenized_i):
                    index_list[idx_tokenized] = idx_original
                    idx_original += 1
            else:
                partitions = produce_all_partitions(tokenized[start_tokenized_i:end_tokenized_i], end_original_i - start_original_i)
                for p in partitions:
                    # find the one
                    fail = False
                    for sub_p, sub_original in zip(p, original[start_original_i:end_original_i]):
                        sub_p_string = "".join(sub_p)
                        sub_p_string = re.escape(sub_p_string)
                        # sub_p_string = sub_p_string.replace("\<unk\>", ".+")  # python 3.6.9
                        sub_p_string = sub_p_string.replace("<unk>", ".+")  # python 3.8.10
                        if not re.match(sub_p_string, sub_original):
                            fail = True
                    if not fail:

                        i_original = start_original_i
                        i_tokenized = start_tokenized_i
                        for sub_list in p:
                            for word in sub_list:
                                index_list[i_tokenized] = i_original
                                i_tokenized += 1
                            i_original += 1
                        break
            w = end_tokenized_i

    assert -1 not in index_list, "Unresolved indices left!"
    mapping = produce_mapping_dic(index_list)
    return mapping, index_list


def flat_column(doc_by_sentence, index):
    flat_column = []
    for sentence in doc_by_sentence:
        for token in sentence:
            flat_column.append(token[index])
    return flat_column


def convert_tags(doc_by_sentence, mapping, index):
    flat_labels = flat_column(doc_by_sentence, index)
    assert len(flat_labels) == len(mapping)

    new_labels = []
    idx_original = sorted(mapping.keys())
    for i in idx_original:
        current_label = flat_labels[i]
        idx_tokenized = mapping[i]
        new_labels.append(current_label)
        #                    1      2      3     4    5
        # map columns info pos, pos_bas, iob, coref, label
        for idx in range(1, len(idx_tokenized)):
            if index in [1, 4]:
                new_labels.append(current_label)
            if index in [2, 5]:
                if current_label.startswith("B-"):
                    new_labels.append(f'I-{current_label[2:]}')
                else:
                    new_labels.append(current_label)
            if index == 3:
                if current_label == "B":
                    new_labels.append("I")
                else:
                    new_labels.append(current_label)
    return new_labels


def structure_document(doc_sentences, mapping, tokenized):
    """
    doc_sentences: [[('Ciba-Geigy', 'nnp', 'B-new', 'B', 'set_1', 'B-new'), () , ], [(),(), ()] ]
    mapping:  {0: [0, 1, 2], 1: [3], 2: [4], 3: [5], 4: [6], 5: [7], ...}
    tokenized: ['<unk>', '-', '<unk>', 'AG', ',', 'the', 'big', 'Swiss', 'chemicals', ...]
    """

    sent_doc = []
    start = 0
    end = 0
    for sent in doc_sentences:
        new_sentence = []
        end += len(sent)
        for i in range(start, end):
            for j in mapping[i]:
                new_sentence.append(tokenized[j])
        start = end
        sent_doc.append(new_sentence)
    return sent_doc


def format_doc(doc, pos, pos_bas, iob, coref, tags, out_file):
    for i in range(len(doc)):
        counter = 0
        for token, part_speech, baseline, iob_orig, entity, label in zip(doc[i], pos[i], pos_bas[i], iob[i], coref[i], tags[i]):
            counter += 1
            out_file.write(f"{str(counter)}\t{token}\t{part_speech}\t{baseline}\t{iob_orig}\t{entity}\t{label}\n")
        out_file.write("\n")


def extract_hidden(data_in, model_name, data_out_tokens, data_out_hidden):

    if model_name == "transfo-xl-wt103":
        tokenizer = TransfoXLTokenizer.from_pretrained(model_name)
        model = TransfoXLModel.from_pretrained(model_name)
    elif model_name.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2Model.from_pretrained(model_name)
    else:
        print(f"Unknown model identifier {model_name}. "
              f"Please specify either transfo-xl-wt103 or a gpt2 variant according to the huggingface identifiers.")
        print("Unable to create output files.")
        exit(1)

    partitions = ["train", "dev", "test"]
    for p in partitions:

        print(f"Extracting files in {p}")
        # load data
        docs_unsplit, docs_sent = load(f"{data_in}/{p}")
        # paths to dump new data
        out_vectors = f"{data_out_hidden}/{model_name}/{p}"
        out_tokens_labels = f"{data_out_tokens}/{model_name}/{p}"

        os.makedirs(out_vectors, exist_ok=True)
        os.makedirs(out_tokens_labels, exist_ok=True)

        for doc in docs_unsplit:
            # remove longer documents to stay within the gpt-2 input length restrictions (1024 tokens),
            # threshold determined to assure same documents being used for different tokenizer versions
            if len(docs_unsplit[doc]) < 800:
                # tokenized
                tokenizer_output = tokenizer(docs_unsplit[doc], is_split_into_words=True, add_special_tokens=False, return_tensors="pt")

                # squeeze to recover tokens
                squeezed = torch.squeeze(tokenizer_output["input_ids"], dim=0)
                tokenizer_words = tokenizer.convert_ids_to_tokens(squeezed)

                # extract and dump last hidden
                outputs = model(tokenizer_output["input_ids"])
                last_hidden_states = outputs.last_hidden_state
                with open(f"{out_vectors}/{doc}", "wb") as vectors_out_file:
                    pickle.dump(last_hidden_states, vectors_out_file)

                # postprocess and map
                clean_original, clean_tokenized = postprocess_tokenized(docs_unsplit[doc], tokenizer_words, model_name)
                mapping_dic, mapping_list = map_versions(clean_original, clean_tokenized, model_name)

                doc_w_sentences = structure_document(docs_sent[doc], mapping_dic, tokenizer_words)

                #                    1      2      3     4    5
                # map columns info pos, pos_bas, iob, coref, label

                converted_pos = convert_tags(docs_sent[doc], mapping_dic, 1)
                pos_w_sentences = structure_document(docs_sent[doc], mapping_dic, converted_pos)

                converted_pos_bas = convert_tags(docs_sent[doc], mapping_dic, 2)
                pos_bas_w_sentences = structure_document(docs_sent[doc], mapping_dic, converted_pos_bas)

                converted_iob = convert_tags(docs_sent[doc], mapping_dic, 3)
                iob_w_sentences = structure_document(docs_sent[doc], mapping_dic, converted_iob)

                converted_coref = convert_tags(docs_sent[doc], mapping_dic, 4)
                coref_w_sentences = structure_document(docs_sent[doc], mapping_dic, converted_coref)

                converted_tags = convert_tags(docs_sent[doc], mapping_dic, 5)
                tags_w_sentences = structure_document(docs_sent[doc], mapping_dic, converted_tags)

                # create and dump new data
                with open(f"{out_tokens_labels}/{doc}", "w", encoding="utf-8") as tokens_out_file:
                    format_doc(doc_w_sentences, pos_w_sentences, pos_bas_w_sentences, iob_w_sentences, coref_w_sentences, tags_w_sentences, tokens_out_file)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Files for extraction of hidden layer')
    parser.add_argument('--data_in', dest='data_in', help='path to ARRAU files', type=str)
    parser.add_argument('--model', dest='model', help='huggingface model identifier', type=str)
    parser.add_argument('--data_out_hidden', dest='data_out_hidden', help='path to out files', type=str)
    parser.add_argument('--data_out_tokens', dest='data_out_tokens', help='path to out files', type=str)
    args = parser.parse_args()

    extract_hidden(args.data_in, args.model, args.data_out_tokens, args.data_out_hidden)
