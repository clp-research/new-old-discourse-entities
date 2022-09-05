# author: Anne Beyer
# date: 13.01.22

# Attention-based model for classifying entities as old or new
# given a discourse context, based on representations extracted
# from the last hidden layer of a pre-trained model

#TODO: create requirements.txt
# pip install comet_ml
# pip install pytorch-nlp

from comet_ml import Experiment  # to monitor experiments on www.comet.ml
import logging
import math
import os
import random
import argparse
import fasttext
import fasttext.util
import yaml
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchnlp.nn import Attention  # pip install pytorch-nlp
from transformers import TransfoXLTokenizer, TransfoXLModel, GPT2Tokenizer, GPT2Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statistics import mean, stdev
from collections import Counter, defaultdict

from classifier_utils import EntityDataset

# Set up logging
logger = logging.getLogger(__name__)
# prepare logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.INFO  # logging.DEBUG, logging.WARN could also be set via argparser
)


class EntityClassifier(nn.Module):
    """
    Attention-based binary classification model
    Predicts status (old/new) of entity in context given
    pre-extracted pre-trained hidden representations
    """
    def __init__(self, dim, ablation):
        super(EntityClassifier, self).__init__()
        self.attention = Attention(dim)
        # attention implementation:
        # https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html#Attention
        #self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(dim, 1)
        # ablation study ignores the context and only performs classification based on the target embeddings
        self.ablation = ablation

    def forward(self, context, target):
        if self.ablation:
            output = target
        else:
            output, _ = self.attention(target, context)  # second output are weights (output = context*weights)
            #output = self.dropout(output)
            #no dropout as we want the model to make use of all the information in the pre-trained embeddings
        output = self.linear(output)
        return output  # see loss_fn below on why not torch.sigmoid(output)


def seed_everything(seed):
    """
    From https://www.kaggle.com/mlwhiz/attention-pytorch-and-keras?scriptVersionId=11337600&cellId=10
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_eos_vector(model):
    """
    Extract <eos> vector representation from pre-trained model
    """
    eos_string = "<eos>"
    if model == "transfo-xl-wt103":
        # load pre-trained tokenizer and model
        tokenizer = TransfoXLTokenizer.from_pretrained(model)
        pretrained_model = TransfoXLModel.from_pretrained(model)
        pretrained_model.eval()
        # extract eos vector
        with torch.no_grad():
            eos_id = tokenizer(eos_string, return_tensors="pt")["input_ids"]
            eos_vector = pretrained_model(eos_id).last_hidden_state
            eos_vector = torch.squeeze(eos_vector, dim=0)  # with removed batch dim
    elif model.startswith("gpt2"):
        eos_string = "<|endoftext|>"
        # load pre-trained tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        pretrained_model = GPT2Model.from_pretrained(model)
        pretrained_model.eval()
        # extract eos vector
        with torch.no_grad():
            eos_id = tokenizer(eos_string, return_tensors="pt")["input_ids"]
            eos_vector = pretrained_model(eos_id).last_hidden_state
            eos_vector = torch.squeeze(eos_vector, dim=0)  # with removed batch dim
    elif model == "fasttext":
        # load pre-trained model
        eos_string = "</s>"
        fasttext.util.download_model('en', if_exists='strict')
        #global ft  # store for later word vector extraction
        ft = fasttext.load_model('cc.en.300.bin')
        # extract eos vector
        eos_vector = ft.get_word_vector(eos_string)
        eos_vector = torch.from_numpy(eos_vector)
        eos_vector = torch.reshape(eos_vector, (1,300))
    else:
        # create random vector
        eos_vector = torch.rand(1,1)
    return eos_vector, eos_string


def train_model(params, device, seed):
    #####################################
    # LOAD DATA
    #####################################
    eos_vector, eos_string = get_eos_vector(params['pre_trained_model'])

    logger.info("Loading training data...")
    train = EntityDataset(params['train_data'], params['train_hidden'], params['pre_trained_model'], eos_vector, eos_string, device, logger)
    logger.info(f"Num samples: {len(train)}")
    logger.info(f"Max context len: {train.max_context_len}")
    experiment.log_image(train.plot_stats("train"))
    train_loader = DataLoader(train, batch_size=params['batch_size'], shuffle=True)

    logger.info("Loading validation data...")
    val = EntityDataset(params['val_data'], params['val_hidden'], params['pre_trained_model'], eos_vector, eos_string, device, logger)
    logger.info(f"Num samples: {len(val)}")
    logger.info(f"Max context len: {val.max_context_len}")
    experiment.log_image(val.plot_stats("val"))
    val_loader = DataLoader(val, batch_size=params['batch_size']*2, shuffle=False)

    #####################################
    # TRAIN
    #####################################
    logger.info("Training...")
    use_context = config['ablation'] if 'ablation' in config else False

    model = EntityClassifier(train.embed_size, use_context)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=params['learning_rate'])
    loss_fn = nn.BCEWithLogitsLoss()  # numerically more stable than
    # using nn.BCELoss() and applying sigmoid to output of model separately according to
    # https://towardsdatascience.com/pytorch-basics-intro-to-dataloaders-and-loss-functions-868e86450047

    # variables for plotting learning curves
    train_losses = []
    val_losses = []

    # parameters for early stopping
    best_epoch_val_loss = math.inf
    patience_counter = 0

    for epoch in tqdm(range(0, params['epochs'])):
        batch_train_losses = []
        batch_val_losses = []
        # training
        # last component is index in train data (to access word representations for debugging)
        for context, target, label, _ in train_loader:
            model.train()
            optimizer.zero_grad()
            prediction = model(context, target)
            prediction = prediction.squeeze(-1)
            loss = loss_fn(prediction, label)
            batch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # validation
        for context, target, label, _ in val_loader:
            model.eval()
            with torch.no_grad():
                prediction = model(context, target)
                prediction = prediction.squeeze(-1)
                val_loss = loss_fn(prediction, label)
                batch_val_losses.append(val_loss.item())

        # every batch loss is already the average of the sample losses in a batch
        epoch_train_loss = sum(batch_train_losses) / len(batch_train_losses)
        epoch_val_loss = sum(batch_val_losses) / len(batch_val_losses)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        logger.info(f"Epoch {epoch} training loss: {epoch_train_loss:.4f}")
        logger.info(f"Epoch {epoch} validation loss: {epoch_val_loss:.4f}")

        # track losses with comet (if 'ignore_comet'=False)
        experiment.log_metric("epoch training loss", epoch_train_loss, step=epoch)
        experiment.log_metric("epoch validation loss", epoch_val_loss, step=epoch)

        # early stopping
        if epoch_val_loss < best_epoch_val_loss:
            # save model
            model_path = os.path.join(params['model'], f"Epochs_{epoch}_seed_{seed}_loss_{epoch_val_loss:.4f}")
            os.makedirs(params['model'], exist_ok=True)
            torch.save(model, model_path)
            logger.info(f"Saved model: {model_path}")
            best_epoch_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            if patience_counter < params['patience']:
                patience_counter += 1
            else:
                break
    # create symlink of model_name to "model_<seed>" in model dir for easier testing
    os.symlink(os.path.abspath(model_path), os.path.join(params['model'], f"model_{seed}"))

    if params['ignore_comet']:
        # plot losses
        plt.plot(train_losses, '-b', label='train_loss')
        plt.plot(val_losses, '-r', label='val_loss')
        plt.xlabel("n epochs")
        plt.legend(loc='upper left')
        plt.title("Loss visualisation")
        plt.savefig(os.path.join(params['model'], "loss_visualisation.png"))


def error_analysis(errors, model_path):
    os.makedirs(model_path, exist_ok=True)
    out_file = os.path.join(model_path, "error_analysis.txt")
    with open(out_file, "w") as error_file:
        num_entites = {0.0: [], 1.0: []}
        misclass_entites = {0.0: [], 1.0: []}
        for label in errors:
            for sample in errors[label]:
                num_entites[label].append(sample[2])
                misclass_entites[label].append(" ".join(sample[1]))
                error_file.write(f"\nTrue label: {label}")
                error_file.write(f"\nContext ({sample[2]} entities):\n{sample[0]}")
                error_file.write(f"\nTarget:\n{sample[1]}")
                error_file.write("\n###############################################")
    logger.info(f"Detailed error analysis saved to {out_file}")

    prons = ["i", "you", "he", "she", "it", "we", "they", "what", "who", "me", "him", "her", "it", "us", "you",
             "them", "whom", "mine", "yours", "his", "hers", "ours", "theirs", "this", "that", "these", "those",
             "who", "whom", "which", "what", "whose", "whoever", "whatever", "whichever", "whomever", "who",
             "whom", "whose", "which", "that", "what", "whatever", "whoever", "whomever", "whichever", "myself",
             "yourself", "himself", "herself", "itself", "ourselves", "themselves"]

    entity_out_file = os.path.join(model_path, "entity_analysis.txt")
    with open(entity_out_file, "w") as entity_file:
        wrong_prons = 0
        wrong_def = 0
        wrong_indef = 0
        for entity in misclass_entites[0.0]:
            if entity.lower() in prons:
                wrong_prons += 1
            elif entity.lower().startswith("the "):
                wrong_def += 1
        for entity in misclass_entites[1.0]:
            if entity.lower().startswith("a ") or entity.lower().startswith("an "):
                wrong_indef += 1
        entity_file.write(f"# wrong pronouns (new instead of old): {wrong_prons}\n")
        entity_file.write(f"# wrong def NPs (new instead of old): {wrong_def}\n")
        entity_file.write(f"# wrong indef NPs (old instead of new): {wrong_indef}\n\n\n")

        wrong_old_counter = Counter(misclass_entites[0.0])
        wrong_new_counter = Counter(misclass_entites[1.0])
        entity_file.write(f"Misclassified old entities as new:\n{wrong_old_counter}\n\n")
        entity_file.write(f"Misclassified new entities as old:\n{wrong_new_counter}\n\n")

    logger.info(f"Detailed entity error analysis saved to {entity_out_file}")

    # plot distribution of misclassifications by number of context entities
    new_count = num_entites[1.0]
    old_count = num_entites[0.0]
    max_entities = max(max(new_count, old_count))
    fig, ax = plt.subplots()
    bins = np.linspace(0, max_entities)
    plt.hist([new_count, old_count], bins, label=[f'misclassified new as old (total {len(new_count)})',
                                                  f'misclassified old as new (total {len(old_count)})'])
    ax.set_xlabel('# Context entities')
    ax.set_ylabel('# Samples')
    ax.set_title('Misclassified samples per number of context entities')
    ax.legend()
    # plt.show()
    plt.savefig(os.path.join(model_path, "error_stats.png"))
    experiment.log_image(os.path.join(model_path, "error_stats.png"))


def test_model(params, device, seed):
    #####################################
    # TEST
    #####################################
    eos_vector, eos_string = get_eos_vector(params['pre_trained_model'])

    logger.info("Testing...")
    logger.info("Loading test data...")
    test = EntityDataset(params['test_data'], params['test_hidden'], params['pre_trained_model'], eos_vector, eos_string, device, logger)
    experiment.log_image(test.plot_stats("test"))
    logger.info(f"Num samples: {len(test)}")
    logger.info(f"Max context len: {test.max_context_len}")
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    if params['pre_trained_model']:
        # load model to evaluate
        model = torch.load(f"{params['model']}_{seed}")
        model.to(device)
        model.eval()
    predictions = []
    gold_labels = []
    errors = {0.0: [], 1.0: []} # save misclassified samples
    with torch.no_grad():
        for context, target, label, index in test_loader:
            if params['pre_trained_model']:
                output = model(context, target)
                prediction_sigmoid = torch.sigmoid(output)  # see model definition and loss_fn in training
                prediction = (prediction_sigmoid.item() > 0.5)
            else:
                # pos-based baseline
                prediction = test.dataset[index]['baseline_label']
                # majority baseline
                # prediction = True
            predictions.append(prediction)
            gold_labels.append(label.item())
            if prediction != label.item():
                errors[label.item()].append((test.dataset[index]['context_words'], test.dataset[index]['target_words'], test.dataset[index]['num_context_entities']))
    logger.info(f"Evaluation: \n{classification_report(gold_labels, predictions)}")
    experiment.log_metric("accuracy", accuracy_score(gold_labels, predictions))
    experiment.log_metric("report", classification_report(gold_labels, predictions))

    if not params['ignore_comet']:
        # Save confusion matrix in comet (if 'ignore_comet'=False)
        cm = confusion_matrix(gold_labels, predictions)
        experiment.log_confusion_matrix(matrix=cm, labels=['old', 'new'])

    error_analysis(errors, os.path.dirname(params['model']))
    return classification_report(gold_labels, predictions, output_dict=True)

#####################################
# MAIN PROCESSING
#####################################


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Entity Classification Experiment')
    parser.add_argument('config_file', help='path to config file', type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    seeds = [42, 123, 9999, 97531, 24680]
    try:

        if config['mode'] == "test":
            old = {"prec": [], "rec": [], "f1": []}
            new = {"prec": [], "rec": [], "f1": []}
            acc = []

        for seed in seeds:
            seed_everything(seed)

            # create experiment with comet
            experiment = Experiment(
                api_key=config['comet_key'],
                project_name=config['comet_project'],
                workspace=config['comet_workspace'],
                disabled=config['ignore_comet'])
            experiment.log_parameters(config)
            experiment.log_parameter("seed", seed)

            if config['mode'] == "train":
                train_model(config, device, seed)

            elif config['mode'] == "test":
                results = test_model(config, device, seed)
                old["prec"].append(results["0.0"]["precision"])
                old["rec"].append(results["0.0"]["recall"])
                old["f1"].append(results["0.0"]["f1-score"])
                new["prec"].append(results["1.0"]["precision"])
                new["rec"].append(results["1.0"]["recall"])
                new["f1"].append(results["1.0"]["f1-score"])
                acc.append(results["accuracy"])
            else:
                logger.error("Config file must specify mode: train|test")
                exit(1)
        if config['mode'] == "test":
            print("new")
            for metric in new:
                print(f"{metric}: {round(mean(new[metric]),2)} ({round(stdev(new[metric]),2)})")
                experiment.log_metric(f"new mean {metric}", mean(new[metric]))
                experiment.log_metric(f"new stdev {metric}", stdev(new[metric]))
            print("old")
            for metric in old:
                print(f"{metric}: {round(mean(old[metric]),2)} ({round(stdev(old[metric]),2)})")
                experiment.log_metric(f"old mean {metric}", mean(old[metric]))
                experiment.log_metric(f"old stdev {metric}", stdev(old[metric]))
            print(f"acc: {round(mean(acc),2)} ({round(stdev(acc),2)})")
            experiment.log_metric("acc mean", mean(acc))
            experiment.log_metric("acc stdev", stdev(acc))
    except KeyError as key:
        logger.error(f"required field {key} not specified in config file {args.config_file}")