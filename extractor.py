"""
Extract data from ARRAU corpus for probing tasks

Syntax:
    $ python3 extractor.py PATH/TO/CORPUS/ [--min_words]

    ex.
    $ python3 extractor.py corpus/by_genre/rts/RST_DTreeBank/

PATH/TO/CORPUS/ needs to contain the annotation directories "[train|dev|test]/MMAX".
--min_words only extracts heads, otherwise spans are extracted.
Creates output files in directories "data/arrau_processed/[heads|spans]/[train|dev|test]/PATH_TO_CORPUS_filename"
Output files are in CoNLL format with the columns:
ID  TOKEN  POS  POS-baseline-label  BIO-label  entity-ID  BIO-old/new-label
"""

import argparse
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


class Span:
    """
    This class is used to sort spans.
    if 2 spans begin at the same index,
    they are compared based on their length:
    the longest span will be sorted first.
    Otherwise they are sorted as normal tuples
    """
    def __init__(self, begin, end):
        self.pair = (begin, end)
        self.begin = begin
        self.end = end

    def __repr__(self):
        return str(self.pair)

    def __lt__(self, other):
        if self.begin != other.begin:
            return self.pair < other.pair
        else:
            return self.end > other.end

    def __iter__(self):
        for element in self.pair:
            yield element


class Project:
    def __init__(self, pathfile):
        path, file = pathfile
        self.path = path
        self.file = file
        self.tokens = []
        self.pos = []
        self.sentences = []
        self.phrases = {}
        self.refs = {}
        self.output = ""
        self.old_new = []
        self.entities = []
        self.iob = []
        self.pos_baseline = [] # def NP & pronouns = old, indef NP = new
        self.rel_objs = []
        self.rel_iob = []
        self.unique_entities = set()

    def collect_tokens(self):
        """
        retrieve tokens from the MMAX project
        """
        words = f"{self.path}/Basedata/{self.file}_words.xml"
        tree = ET.parse(words)
        root = tree.getroot()
        for child in root:
            self.tokens.append(child.text)

    def collect_pos(self):
        """
        retrieve POS tags from the MMAX project
        """
        pos = f"{self.path}/markables/{self.file}_pos_level.xml"
        tree = ET.parse(pos)
        root = tree.getroot()
        for child in root:
            self.pos.append(child.attrib["tag"])

    def collect_sentences(self):
        """
        retrieve sentences from the MMAX project
        """
        sentence = f"{self.path}/markables/{self.file}_sentence_level.xml"
        tree = ET.parse(sentence)
        root = tree.getroot()

        for child in root:
            span = child.attrib["span"]
            r = re.findall("[0-9]+", span)
            if len(r) == 1:
                sen = (int(r[0])-1, int(r[0]))
            else:
                sen = (int(r[0])-1, int(r[1]))

            self.sentences.append(sen)

    def collect_phrases(self):
        coref = f"{self.path}/markables/{self.file}_phrase_level.xml"
        tree = ET.parse(coref)
        root = tree.getroot()
        for child in root:
            span = child.attrib["span"]
            r = re.findall("[0-9]+", span)

            if len(r) == 1:
                span = (int(r[0]), int(r[0]))
            else:
                span = (int(r[0]), int(r[1]))

            self.phrases[child.attrib["id"]] = {
                "span": span,
                "new_old": child.attrib["reference"]
            }

    def collect_refs(self):
        """
        retrieve references from the MMAX project
        saved as tuples of indices
        """
        coref = f"{self.path}/markables/{self.file}_coref_level.xml"
        tree = ET.parse(coref)
        root = tree.getroot()
        for child in root:
            span = child.attrib["span"]
            r = re.findall(r"[0-9]+", span)
            if len(r) == 1:
                span = Span(int(r[0])-1, int(r[0]))
            else:
                span = Span(int(r[0])-1, int(r[1]))

            self.refs[span] = {
                "reference": child.attrib["reference"],
                "set": child.attrib["coref_set"]
            }

            if span.begin == span.end - 1:
                min_word = span.begin
                self.refs[span]["min_word"] = Span(min_word, min_word + 1)

            elif "min_ids" in child.attrib:
                r = re.findall(r"[0-9]+", child.attrib["min_ids"])
                if len(r) == 1:
                    min_word = Span(int(r[0])-1, int(r[0]))
                else:
                    min_word = Span(int(r[0])-1, int(r[1]))

                self.refs[span]["min_word"] = min_word

            if (("related_object" in child.attrib) and
                    child.attrib["related_object"] == "yes"):
                related_phrase = child.attrib["related_phrase"]
                self.refs[span]["related_object"] = related_phrase

    def generate_output(self):
        """
        calculates the number of unique entities for each document
        """
        sen_refs = set()
        ref_number = 0
        for Is, Es in self.sentences:
            for index, reference in self.refs.items():
                Ir, Er = index
                if Ir >= Is and Er <= Es:
                    if reference["reference"] != "non_referring":
                        ref_number += 1
                        sen_refs.add(reference["set"])

            str_sen = " ".join(self.tokens[Is:Es])
            self.output += f"{str_sen}\t{ref_number}\t{len(sen_refs)}\n"

    def save(self, path):
        if self.output != "":
            filename = self.path.replace("/", "__")
            filename += f"__{self.file}"

            outputfile = Path(f"{path}/{filename}")

            with open(outputfile, "w", encoding="utf-8") as ofile:
                ofile.write(self.output)

    def generate_lists(self, min_words):
        """
        based on token position, this function will create:
            - entity list
            - old/new column
            - iob column
            - related object column
            - pos_baseline column
        """
        entity_list = [[] for _ in range(len(self.tokens))]
        old_new = [[] for _ in range(len(self.tokens))]
        iob = [[] for _ in range(len(self.tokens))]
        pos_baseline = [[] for _ in range(len(self.tokens))]

        for span, coref_type in sorted(self.refs.items()):
            # ignore non_referring references
            if coref_type["reference"] != "non_referring":
                if min_words:
                    # extract only head span
                    # ignore span if no min_word is annotated
                    if "min_word" in coref_type:
                        begin, end = coref_type["min_word"]
                    else:
                        continue  # todo: convert to error?
                else:
                    begin, end = span

                # combine embedded mentions
                if all(entity_list[i] == [] for i in range(begin, end)):
                    first = True
                    baseline_tag = self.get_baseline_prediction(span.begin) # get baseline label from pos tag of first token

                    for i in range(begin, end):
                        # save old/new tag
                        if coref_type["reference"] == "new":
                            tag = "B-new"
                        else:
                            tag = "B-old"

                        if first is True:
                            old_new[i].append(tag)
                            iob[i].append("B")
                            pos_baseline[i].append(baseline_tag)
                            first = False
                        else:
                            if coref_type["reference"] == "new":
                                old_new[i].append("I-new")
                            else:
                                old_new[i].append("I-old")
                            iob[i].append("I")
                            if "new" in baseline_tag:
                                pos_baseline[i].append("I-new")
                            else:
                                pos_baseline[i].append("I-old")

                        # save set
                        entity_list[i].append(coref_type["set"])
                        self.unique_entities.add(coref_type["set"])

        self.old_new = old_new
        self.entities = entity_list
        self.iob = iob
        self.pos_baseline = pos_baseline

    def get_baseline_prediction(self, i):
        """
        return baseline prediction based on pos tag + token at position i
        Assumption: def NP/ pronoun = old, else = new
        """
        if self.pos[i] == "prp":
            return "B-old"
        if self.pos[i] == "dt":
            if self.tokens[i].lower() in ["the", "that", "these", "those", "this"]:
                return "B-old"
        return "B-new"

    def save_iob(self, path):
        """
        save the data in a tab separated file with columns:
            id  token  pos  pos_baseline_tag  simple_iob  reference_id  old/new-tag

        the name of the file is the path of the original file
        with / substituted by __
        The number of total unique entities is saved in the
        filename after : at the very end
        """
        # make sure path exists
        os.makedirs(Path(path), exist_ok=True)

        filename = re.findall(r"\w+.*", self.path)
        filename = "".join(filename)
        filename = filename.replace("/", "__")
        filename += f"__{self.file}"
        unique_entities = len(self.unique_entities)

        outputfile = Path(f"{path}/{filename}:{unique_entities}")
        sent_bound = set([i[1] for i in self.sentences])

        with open(outputfile, "w", encoding="utf-8") as ofile:
            for i in range(len(self.tokens)):

                if i+1 in sent_bound:
                    newline = "\n"
                else:
                    newline = ""

                token = self.tokens[i]
                pos = self.pos[i]

                if self.pos_baseline[i] == []:
                    pos_baseline = "O"
                else:
                    pos_baseline = self.pos_baseline[i][0]

                if self.iob[i] == []:
                    iob = "O"
                    entity = "O"
                    old_new = "O"
                else:
                    # can be more than one for nested entities (if not ignored)
                    iob = "|".join(self.iob[i])
                    entity = "|".join(self.entities[i])
                    old_new = "|".join(self.old_new[i])

                # save in format:
                #    id  token  pos  pos_baseline_tag  simple_iob  reference_id  old/new-tag
                ofile.write(
                    f"{i+1}\t{token}\t"
                    f"{pos}\t{pos_baseline}\t"
                    f"{iob}\t"
                    f"{entity}\t"
                    f"{old_new}")

                ofile.write(f"{newline}\n")

    def process(self, min_words):
        """
        process a document:
        since not all tokens are annotated, the try block will
        try to collect all entities needed for the extraction.
        If the collection was succesfull, the entity lists
        will be generated
        """
        try:
            self.collect_tokens()
            self.collect_pos()
            self.collect_sentences()
            self.collect_phrases()
            self.collect_refs()
            collected = True
        except KeyError as e:
            # some markables have no "coref_set" annotation in the _coref-level.xml
            print(f"Invalid file: {self.path}/{self.file}: missing annotations")
            print(e)
            collected = False

        if collected:
            self.generate_lists(min_words)
        return collected


def retrieve_candidates(path):
    """
    look for .mmax files checks if they are valid aka.
    they have a sentence, pos level and coref level.
    If so, return a list of tuples (path, filename)
    """
    endings = ["_sentence_level.xml", "_pos_level.xml", "_coref_level.xml"]
    candidates = []
    for path, _, files in os.walk(path):
        for file in files:
            if file[-5:] == ".mmax":
                all_endings = True
                for ending in endings:
                    filepath = Path(f"{path}/markables/{file[:-5]}{ending}")
                    if not os.path.isfile(filepath):
                        all_endings = False

                if all_endings:
                    candidates.append((path, file[:-5]))

    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", metavar="PATH", help="Path to the corpus")
    parser.add_argument("--min_words", action="store_true", help="only annotate the min_words (heads)")
    args = parser.parse_args()

    parent_dir = Path("arrau_processed")

    for split in ["train", "dev", "test"]:
        path = os.path.join(os.path.join(args.path,split),"MMAX")
        if not os.path.isdir(path):
            print(f"Corpus path {path} doesn't exist. Skipping it.")
        else:
            # retrieve candidates and process files
            candidates = retrieve_candidates(path)

            for pathfile in candidates:
                prj = Project(pathfile)
                success = prj.process(args.min_words)

                if success:
                    if args.min_words:
                        output_folder = f"data/arrau_processed/heads/{split}"
                    else:
                        output_folder = f"data/arrau_processed/spans/{split}"

                    # make sure output directory exists
                    os.makedirs(Path(output_folder), exist_ok=True)
                    prj.save_iob(Path(output_folder))
