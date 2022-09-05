# run sequence labelling probing experiments

import math
import os
import argparse
import yaml
import random
import torch
import logging
import pickle

from utils import Vocabulary, EntityDataset
from NER_model_wCRF import EntityLabeler
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, LBFGS

import matplotlib.pyplot as plt
from seqeval.metrics import classification_report, accuracy_score
from statistics import mean, stdev


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.INFO  # logging.DEBUG, logging.WARN could also be set via argparser
)


def train(config, device, seed, train_data, val_data):
    torch.autograd.set_detect_anomaly(True)

    """ Run model training """
    logger.info("***** Running training *****")

    # prepare data for training and validation
    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    vocab_size = train_data.vocab_size if not config["pre_trained"] else None  # only relevant if no pre-trained model is used

    val_dataloader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    # prepare model
    model = EntityLabeler(vocab_size, train_data.label_size, config)
    model.to(device)

    # optimizer
    optimizer = Adam(model.parameters(), lr=config["lr"])

    # parameters for early stopping
    best_epoch_loss = math.inf
    patience = 3
    patience_counter = 0

    # variables for plotting learning curves
    train_losses = []
    val_losses = []

    for epoch in range(0, config["epochs"]):
        batch_train_losses = []
        batch_val_losses = []
        # training
        for sequence, labels, masks, words in train_dataloader:
            model.train()
            optimizer.zero_grad()
            loss = model(sequence, labels, masks) ## it has its own loss computation
            batch_train_losses.append(loss.item()) ## summed log-likelihoods of each sequence in the batch
            loss.backward()
            optimizer.step()

        # validate every epoch
        for sequence, labels, masks, words in val_dataloader:
            model.eval()
            with torch.no_grad():
                loss = model(sequence, labels, masks) ## it has its own loss computation
            batch_val_losses.append(loss.item()) ## summed log-likelihoods of each sequence in the batch

        epoch_train_loss = sum(batch_train_losses) / (len(batch_train_losses) * config["batch_size"])
        epoch_val_loss = sum(batch_val_losses) / (len(batch_val_losses) * config["batch_size"])

        # track losses with comet (if 'ignore_comet'=False)
        # experiment.log_metric("epoch training loss", epoch_train_loss, step=epoch)
        # experiment.log_metric("epoch validation loss", epoch_val_loss, step=epoch)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        logger.info(f"Epoch {epoch} training loss: {epoch_train_loss:.4f}")
        logger.info(f"Epoch {epoch} validation loss: {epoch_val_loss:.4f}")

        # early stopping
        if epoch_val_loss < best_epoch_loss:
            model_name = f"Epo_{epoch}_see_{seed}_bat_{config['batch_size']}_hid_{config['hidden']}_los_{epoch_val_loss:.4f}"
            model_path = os.path.join(config['model'], model_name)
            # save model
            os.makedirs(config["model"], exist_ok=True)
            torch.save(model, model_path)
            logger.info(f"Saved model: {model_path}")
            best_epoch_loss = epoch_val_loss
            patience_counter = 0
        else:
            if patience_counter < patience:
                patience_counter += 1
            else:
                break
    # create symlink of model_name to "model_<seed>" in model dir for easier testing
    os.symlink(os.path.abspath(model_path), os.path.join(config['model'], f"model_{seed}"))

    #plot losses
    plt.plot(train_losses, '-b', label='train_loss')
    plt.plot(val_losses, '-r', label='val_loss')
    plt.xlabel("n epochs")
    plt.legend(loc='upper left')
    plt.title("Loss visualisation")
    plt.savefig(os.path.join(config['model'], f"model_{seed}_visualisation.png"))

    # -----------------------------------------------------------------------


def eval(config, device, seed, test_data):

    """ Run model evaluation """
    logger.info("***** Running evaluation *****")

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    if "baseline" not in config:
        # declare model and load weights to evaluate
        model = torch.load(f"{config['model']}_{seed}")
        model.to(device)
        model.eval()

    # evaluate
    gold_labels = []
    pred_labels = []
    os.makedirs(config["log_dir"], exist_ok=True)
    with open(os.path.join(config["log_dir"], f"{seed}_model_predictions.log"), "w") as pred_file:
        with torch.no_grad():
            if "baseline" in config:
                for words, labels_golds, baseline_labels in test_dataloader:
                    labels_golds = [l[0] for l in labels_golds]
                    baseline_labels = [b[0] for b in baseline_labels]
                    # save predictions for manual evaluation
                    pred_file.write(f"{words}\n{labels_golds}\n{baseline_labels}\n\n")
                    gold_labels.append(labels_golds)
                    pred_labels.append(baseline_labels)
            else:
                for hidden, labels, masks, words in test_dataloader:
                    # expects: (batch_size, seq_length, num_tags)
                    preds = model(hidden)
                    #print("sequence shape ==> ", sequence.shape)
                    #print("preds shape ==> ", preds)
                    labels = labels.reshape(labels.shape[0] * labels.shape[1])
                    #print("labels shape ==> ", labels.shape)
                    # transform labels
                    labels_golds = [test_data.id_to_label[l] for l in labels.cpu().numpy().tolist()]
                    labels_preds = [test_data.id_to_label[l] for l in preds[0]]
                    # preds[0] because function returns the result as a list of lists (not a tensor),
                    # corresponding to a matrix of shape (n_sentences, max_len)

                    # save predictions for manual evaluation
                    pred_file.write(f"{words}\n{labels_golds}\n{labels_preds}\n\n")

                    gold_labels.append(labels_golds)
                    pred_labels.append(labels_preds)

    logger.info(f"data_size: {len(test_data)}")
    logger.info(f"Accuracy: {accuracy_score(gold_labels, pred_labels)}")
    logger.info(f"Detailed evaluation:\n{classification_report(gold_labels, pred_labels)}")
    return classification_report(gold_labels, pred_labels, output_dict=True)
    # -------------------------------------------------


def overfit(config, device):

    """ Overfit model on a single batch """
    logger.info("***** Running overfit *****")

    # prepare data

    # create vocabulary
    vocab = Vocabulary()
    data = EntityDataset("train", config, device, logger, vocab, padding=True)

    vocab_size = data.vocab_size if not config["pre_trained"] else None # only relevant if no pre-trained model is used
    label_size = data.label_size

    dataloader = DataLoader(data, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    # use just one batch for overfitting
    sequence, labels, masks, words = next(iter(dataloader))

    # prepare model
    model = EntityLabeler(vocab_size, label_size, config)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)

    for epoch in range(0, config["epochs"]):
        model.train()
        loss = model(sequence, labels, masks)
        epoch_loss = loss.item()
        optimizer.zero_grad()
        loss.backward(retain_graph=True) # used following
        # https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
        optimizer.step()
        logger.info(f"Epoch {epoch} training loss: {epoch_loss / config['batch_size']:.4f}")

    # -----------------------------------------------------------------

def seed_everything(seed):
    """
    From https://www.kaggle.com/mlwhiz/attention-pytorch-and-keras?scriptVersionId=11337600&cellId=10
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# -----------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Experiment Args')
    parser.add_argument('config_file', help='path to config file')
    parser.add_argument('--CPU', dest='CPU', help='use CPU instead of GPU', action='store_true')
    parser.add_argument('--DEBUG', dest='DEBUG', help='enter debug mode', action='store_true')
    args = parser.parse_args()

    # load config
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # set device
    device = torch.device("cuda" if not args.CPU and torch.cuda.is_available() else "cpu")

    seeds = [170841, 28202, 335628, 369, 456889]

    if config['mode'] == "test":
        # load vocabulary
        vocab = None
        if not config["pre_trained"]:
            with open(f"{config['model']}_vocab.pkl", "rb") as vocab_f:
                vocab = pickle.load(vocab_f)
        # prepare data
        test_data = EntityDataset("test", config, device, logger, vocab, padding=False)
        # prepare stats
        old = {"prec": [], "rec": [], "f1": []}
        new = {"prec": [], "rec": [], "f1": []}
        avgf1 = []

    if config['mode'] == "train":
        # create vocabulary
        vocab = Vocabulary()
        # prepare data for training and validation
        train_data = EntityDataset("train", config, device, logger, vocab, padding=True)
        val_data = EntityDataset("dev", config, device, logger, vocab, padding=True)

        # save vocab
        if not config["pre_trained"]:
            os.makedirs(config["model"], exist_ok=True)
            with open(os.path.join(config['model'], f"model_vocab.pkl"), "wb") as vocab_f:
                pickle.dump(vocab, vocab_f)

    for seed in seeds:
        seed_everything(seed)
        #logger.info((f"Model seed:{seed}"))

        # create experiment with comet
        # experiment = Experiment(
        #     api_key=config['comet_key'],
        #     project_name=config['comet_project'],
        #     workspace=config['comet_workspace'],
        #     disabled=config['ignore_comet'])
        # experiment.log_parameters(config)
        # experiment.log_parameter("seed", seed)

        if config['mode'] == "train":
            logger.info('Starting training mode...')
            train(config, device, seed, train_data, val_data)

        elif config['mode'] == "test":
            logger.info('Starting evaluation mode with test...')
            results = eval(config, device, seed, test_data)
            print(results)
            #results = test_model(config, device, seed)
            old["prec"].append(results["old"]["precision"])
            old["rec"].append(results["old"]["recall"])
            old["f1"].append(results["old"]["f1-score"])
            new["prec"].append(results["new"]["precision"])
            new["rec"].append(results["new"]["recall"])
            new["f1"].append(results["new"]["f1-score"])
            avgf1.append(results["weighted avg"]["f1-score"])
        else:
            logger.error("Config file must specify mode: train|test")
            exit(1)
    if config['mode'] == "test":
        print("new")
        for metric in new:
            print(f"{metric}: {mean(new[metric]):.2f} ({stdev(new[metric]):.2f})")
            # experiment.log_metric(f"new mean {metric}", mean(new[metric]))
            # experiment.log_metric(f"new stdev {metric}", stdev(new[metric]))
        print("old")
        for metric in old:
            print(f"{metric}: {mean(old[metric]):.2f} ({stdev(old[metric]):.2f})")
            # experiment.log_metric(f"old mean {metric}", mean(old[metric]))
            # experiment.log_metric(f"old stdev {metric}", stdev(old[metric]))
        print(f"avg f1: {mean(avgf1):.2f} ({stdev(avgf1):.2f})")
        # experiment.log_metric("avg f1 mean", mean(avgf1))
        # experiment.log_metric("avg f1 stdev", stdev(avgf1))
