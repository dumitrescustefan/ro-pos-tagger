import os, sys, json, torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np


class TransformerModel(pl.LightningModule):
    def __init__(self, model_name="dumitrescustefan/bert-base-romanian-cased-v1", upos_tag_list=[],
            xpos_tag_list=[], lr=2e-05, model_max_length=512):
        super().__init__()

        print("Loading AutoModel [{}]...".format(model_name))
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.upos_layer = torch.nn.Linear(768, len(upos_tag_list))
        self.xpos_layer = torch.nn.Linear(768, len(xpos_tag_list))
        self.dropout = nn.Dropout(0.1)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.lr = lr
        self.model_max_length = model_max_length
        self.upos_tag_list = upos_tag_list
        self.xpos_tag_list = xpos_tag_list

        self.train_data = {"gold_upos":[], "pred_upos":[], "gold_xpos":[], "pred_xpos":[], "loss":[]}
        self.valid_data = {"gold_upos":[], "pred_upos":[], "gold_xpos":[], "pred_xpos":[], "loss":[]}
        self.test_data = {"gold_upos":[], "pred_upos":[], "gold_xpos":[], "pred_xpos":[], "loss":[]}

    def forward(self, input_ids, attention_mask):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = self.dropout(output["last_hidden_state"]) # [bs, seq_len, model dim]
        upos = self.upos_layer(logits) # [bs, seq_len, upos_len]
        xpos = self.xpos_layer(logits) # [bs, seq_len, xpos_len]

        return upos, xpos

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention = batch["attention"]
        gold_upos = batch["upos"]
        gold_xpos = batch["xpos"]
        token_idx = batch["token_idx"]

        logits_upos, logits_xpos = self(input_ids, attention)  # [batch_size, seq_len, upox/xpos one-hot]

        batch_size = logits_upos.size(0)

        pred_upos = torch.argmax(logits_upos, dim=-1)  # reduce to [batch_size, seq_len]
        pred_xpos = torch.argmax(logits_xpos, dim=-1)  # reduce to [batch_size, seq_len]

        loss_upos = self.loss(logits_upos.view(-1, len(self.upos_tag_list)), gold_upos.view(-1))
        loss_xpos = self.loss(logits_xpos.view(-1, len(self.xpos_tag_list)), gold_xpos.view(-1))
        loss = 0.5 * loss_upos + 0.5 * loss_xpos

        gold_upos = batch["upos"].detach().cpu().tolist()
        gold_xpos = batch["xpos"].detach().cpu().tolist()
        pred_upos = pred_upos.detach().cpu().tolist()
        pred_xpos = pred_xpos.detach().cpu().tolist()
        token_idx = token_idx.detach().cpu().tolist()

        for batch_idx in range(batch_size):  # for each sentence
            sentence_gold_upos = gold_upos[batch_idx]
            sentence_gold_xpos = gold_xpos[batch_idx]
            sentence_token_idx = token_idx[batch_idx]
            sentence_pred_upos = pred_upos[batch_idx]
            sentence_pred_xpos = pred_xpos[batch_idx]
            for i in range(0, max(sentence_token_idx) + 1):
                pos = sentence_token_idx.index(i)  # find next token index and get pred and gold
                self.train_data["gold_upos"].append(sentence_gold_upos[pos])
                self.train_data["pred_upos"].append(sentence_pred_upos[pos])
                self.train_data["gold_xpos"].append(sentence_gold_xpos[pos])
                self.train_data["pred_xpos"].append(sentence_pred_xpos[pos])

        self.train_data["loss"].append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.log("train/loss", sum(self.train_data["loss"]) / len(self.train_data["loss"]), prog_bar=True)
        self.log("train/upos_f1", f1_score(self.train_data["gold_upos"], self.train_data["pred_upos"], average="micro"))
        self.log("train/upos_precision", precision_score(self.train_data["gold_upos"], self.train_data["pred_upos"], average="micro"))
        self.log("train/upos_recall", recall_score(self.train_data["gold_upos"], self.train_data["pred_upos"], average="micro"))
        self.log("train/xpos_f1", f1_score(self.train_data["gold_xpos"], self.train_data["pred_xpos"], average="micro"))
        self.log("train/xpos_precision", precision_score(self.train_data["gold_xpos"], self.train_data["pred_xpos"], average="micro"))
        self.log("train/xpos_recall", recall_score(self.train_data["gold_xpos"], self.train_data["pred_xpos"], average="micro"))

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention = batch["attention"]
        gold_upos = batch["upos"]
        gold_xpos = batch["xpos"]
        token_idx = batch["token_idx"]

        logits_upos, logits_xpos = self(input_ids, attention)  # [batch_size, seq_len, upox/xpos one-hot]

        batch_size = logits_upos.size(0)

        pred_upos = torch.argmax(logits_upos, dim=-1) # reduce to [batch_size, seq_len]
        pred_xpos = torch.argmax(logits_xpos, dim=-1)  # reduce to [batch_size, seq_len]

        loss_upos = self.loss(logits_upos.view(-1, len(self.upos_tag_list)), gold_upos.view(-1))
        loss_xpos = self.loss(logits_xpos.view(-1, len(self.xpos_tag_list)), gold_xpos.view(-1))
        loss = 0.5 * loss_upos + 0.5 * loss_xpos

        gold_upos = batch["upos"].detach().cpu().tolist()
        gold_xpos = batch["xpos"].detach().cpu().tolist()
        pred_upos = pred_upos.detach().cpu().tolist()
        pred_xpos = pred_xpos.detach().cpu().tolist()
        token_idx = token_idx.detach().cpu().tolist()

        for batch_idx in range(batch_size): # for each sentence
            sentence_gold_upos = gold_upos[batch_idx]
            sentence_gold_xpos = gold_xpos[batch_idx]
            sentence_token_idx = token_idx[batch_idx]
            sentence_pred_upos = pred_upos[batch_idx]
            sentence_pred_xpos = pred_xpos[batch_idx]
            for i in range(0, max(sentence_token_idx)+1):
                pos = sentence_token_idx.index(i) # find next token index and get pred and gold
                self.valid_data["gold_upos"].append(sentence_gold_upos[pos])
                self.valid_data["pred_upos"].append(sentence_pred_upos[pos])
                self.valid_data["gold_xpos"].append(sentence_gold_xpos[pos])
                self.valid_data["pred_xpos"].append(sentence_pred_xpos[pos])

        self.valid_data["loss"].append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        upos_f1 = f1_score(self.valid_data["gold_upos"], self.valid_data["pred_upos"], average="micro")
        xpos_f1 = f1_score(self.valid_data["gold_xpos"], self.valid_data["pred_xpos"], average="micro")
        self.log("valid/loss", sum(self.valid_data["loss"]) / len(self.valid_data["loss"]), prog_bar=True)
        self.log("valid/upos_f1", upos_f1)
        self.log("valid/upos_precision", precision_score(self.valid_data["gold_upos"], self.valid_data["pred_upos"], average="micro"))
        self.log("valid/upos_recall", recall_score(self.valid_data["gold_upos"], self.valid_data["pred_upos"], average="micro"))
        self.log("valid/xpos_f1", xpos_f1)
        self.log("valid/xpos_precision", precision_score(self.valid_data["gold_xpos"], self.valid_data["pred_xpos"], average="micro"))
        self.log("valid/xpos_recall", recall_score(self.valid_data["gold_xpos"], self.valid_data["pred_xpos"], average="micro"))

        self.log("valid/join_accuracy", (upos_f1+xpos_f1)/2.)

        print("\n Validation results: ")
        print(f"\t UPOS f1 = {upos_f1:.4f}\tXPOS f1 = {xpos_f1:.4f}")

        self.valid_data = {"gold_upos":[], "pred_upos":[], "gold_xpos":[], "pred_xpos":[], "loss":[]}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention = batch["attention"]
        gold_upos = batch["upos"]
        gold_xpos = batch["xpos"]
        token_idx = batch["token_idx"]

        logits_upos, logits_xpos = self(input_ids, attention)  # [batch_size, seq_len, upox/xpos one-hot]

        batch_size = logits_upos.size(0)

        pred_upos = torch.argmax(logits_upos, dim=-1)  # reduce to [batch_size, seq_len]
        pred_xpos = torch.argmax(logits_xpos, dim=-1)  # reduce to [batch_size, seq_len]

        loss_upos = self.loss(logits_upos.view(-1, len(self.upos_tag_list)), gold_upos.view(-1))
        loss_xpos = self.loss(logits_xpos.view(-1, len(self.xpos_tag_list)), gold_xpos.view(-1))
        loss = 0.5 * loss_upos + 0.5 * loss_xpos

        gold_upos = batch["upos"].detach().cpu().tolist()
        gold_xpos = batch["xpos"].detach().cpu().tolist()
        pred_upos = pred_upos.detach().cpu().tolist()
        pred_xpos = pred_xpos.detach().cpu().tolist()
        token_idx = token_idx.detach().cpu().tolist()

        for batch_idx in range(batch_size):  # for each sentence
            sentence_gold_upos = gold_upos[batch_idx]
            sentence_gold_xpos = gold_xpos[batch_idx]
            sentence_token_idx = token_idx[batch_idx]
            sentence_pred_upos = pred_upos[batch_idx]
            sentence_pred_xpos = pred_xpos[batch_idx]
            for i in range(0, max(sentence_token_idx) + 1):
                pos = sentence_token_idx.index(i)  # find next token index and get pred and gold
                self.test_data["gold_upos"].append(sentence_gold_upos[pos])
                self.test_data["pred_upos"].append(sentence_pred_upos[pos])
                self.test_data["gold_xpos"].append(sentence_gold_xpos[pos])
                self.test_data["pred_xpos"].append(sentence_pred_xpos[pos])

        self.test_data["loss"].append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def test_epoch_end(self, outputs):
        self.log("test/loss", sum(self.test_data["loss"]) / len(self.test_data["loss"]), prog_bar=True)
        self.log("test/upos_f1", f1_score(self.test_data["gold_upos"], self.test_data["pred_upos"], average="micro"))
        self.log("test/upos_precision", precision_score(self.test_data["gold_upos"], self.test_data["pred_upos"], average="micro"))
        self.log("test/upos_recall", recall_score(self.test_data["gold_upos"], self.test_data["pred_upos"], average="micro"))
        self.log("test/xpos_f1", f1_score(self.test_data["gold_xpos"], self.test_data["pred_xpos"], average="micro"))
        self.log("test/xpos_precision", precision_score(self.test_data["gold_xpos"], self.test_data["pred_xpos"], average="micro"))
        self.log("test/xpos_recall", recall_score(self.test_data["gold_xpos"], self.test_data["pred_xpos"], average="micro"))

        self.test_data = {"gold_upos": [], "pred_upos": [], "gold_xpos": [], "pred_xpos": [], "loss": []}

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)


class MyDataset(Dataset):
    def __init__(self, file):
        self.sentences = []
        with open(file, "r", encoding="utf8") as f:
            sentence = []
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if len(line) == 0:
                    if len(sentence)>0: # finish instance
                        self.sentences.append(sentence)
                        sentence = []
                else: # add to instance
                    parts = line.split()
                    sentence.append({"word":parts[1], "upos":parts[3], "xpos":parts[4], "space_after": True if parts[9]!="_" else False})
        print(f"\t File {file} contains {len(self.sentences)} sentences.")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]

class MyCollator(object):
    def __init__(self, tokenizer, upos_tag_list, xpos_tag_list):
        self.tokenizer = tokenizer
        self.upos_tag_list = upos_tag_list
        self.xpos_tag_list = xpos_tag_list
        try:
            self.pad = tokenizer.pad_id
        except:
            self.pad = 0
            print(f"Warning, tokenizer has no pad token, using default 0 index!")

    def __call__(self, input_batch):
        """
        input_batch is a list of batch_size sentences from the dataloader with word/upos/xpos/space_after elements
        output should be a dict of tensors like:
        {
            "input_ids": [bs, padded_input_ids]
            "attention": [bs, padded_attention] same size as input_ids
            "upos": [bs, padded_input_ids] of upos indices
            "xpos": [bs, padded_input_ids] of xpos indices
            "token_idx": [bs, padded_input_ids] with the original token it belongs to
        }
        """
        batch_input_ids, batch_attention, batch_upos, batch_xpos, batch_token_idx = [], [], [], [], []
        max_len = 0

        for sentence in input_batch:
            sentence_input_ids, sentence_attention, sentence_upos, sentence_xpos, sentence_token_idx = [], [], [], [], []

            for word_idx, elem in enumerate(sentence):
                # tokenize word
                subids = self.tokenizer.encode(elem["word"], add_special_tokens=False)
                sentence_input_ids.extend(subids)

                # upos and xpos indices
                sentence_upos.extend([self.upos_tag_list.index(elem["upos"])] * len(subids))
                sentence_xpos.extend([self.xpos_tag_list.index(elem["xpos"])] * len(subids))

                # token_index
                sentence_token_idx.extend([word_idx]*len(subids))

            # sentence-level attention
            sentence_attention = [1]*len(sentence_input_ids)

            # checks at sentence-level and update max_len
            assert len(sentence_input_ids) == len(sentence_upos)
            assert len(sentence_input_ids) < 510
            max_len = max(max_len, len(sentence_input_ids))

            # save results of this sentence
            batch_input_ids.append(sentence_input_ids)
            batch_attention.append(sentence_attention)
            batch_upos.append(sentence_upos)
            batch_xpos.append(sentence_xpos)
            batch_token_idx.append(sentence_token_idx)

        # prepend and append special tokens, if needed
        """instance_ids = [self.tokenizer.cls_token_id] + instance_ids + [self.tokenizer.sep_token_id]
        instance_labels = [0] + instance_labels + [0]
        instance_attention = [1] * len(instance_ids)
        instance_token_idx = [-1] + instance_token_idx # no need to pad the last, will do so automatically at return
        """

        # batch-level padding
        padded_input_ids, padded_attention, padded_upos, padded_xpos, padded_token_idx = [], [], [], [], []
        for i in range(len(batch_input_ids)):
            sentence_input_ids = batch_input_ids[i]
            sentence_attention = batch_attention[i]
            sentence_upos = batch_upos[i]
            sentence_xpos = batch_xpos[i]
            sentence_token_idx = batch_token_idx[i]

            pad_len = max_len - len(sentence_input_ids)

            sentence_input_ids.extend([self.pad] * pad_len) # pad token TODO
            sentence_attention.extend([0] * pad_len)
            sentence_upos.extend([-1] * pad_len)
            sentence_xpos.extend([-1] * pad_len)
            sentence_token_idx.extend([-1] * pad_len)

            padded_input_ids.append(sentence_input_ids)
            padded_attention.append(sentence_attention)
            padded_upos.append(sentence_upos)
            padded_xpos.append(sentence_xpos)
            padded_token_idx.append(sentence_token_idx)

        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention": torch.tensor(padded_attention),
            "upos": torch.tensor(padded_upos),
            "xpos": torch.tensor(padded_xpos),
            "token_idx": torch.tensor(padded_token_idx)
        }

def run_evaluation(
        automodel_name: str,
        tokenizer_name: str,
        model_max_length: int,

        train_file: str = None,
        validation_file: str = None,
        test_file: str = None,

        gpus: int = 1,
        batch_size: int = 32,
        accumulate_grad_batches: int = 1,
        lr: float = 3e-5,

        experiment_iterations: int = 1,
        results_file: str = "results_ronec_v2.json",
        save_model = False
    ):

    print(f"Running {experiment_iterations} experiments with model {automodel_name}")

    print("\t batch size is {}, accumulate grad batches is {}, final batch_size is {}\n".format(
        batch_size,
        accumulate_grad_batches,
        batch_size * accumulate_grad_batches)
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Loading data...")
    train_dataset = MyDataset(train_file)
    val_dataset = MyDataset(validation_file)
    test_dataset = MyDataset(test_file)

    # collect tag set for upos and xpos
    uposes, xposes = set(), set()
    for sentence in train_dataset.sentences + val_dataset.sentences + test_dataset.sentences:
        for word in sentence:
            uposes.add(word["upos"])
            xposes.add(word["xpos"])
    upos_tag_list = sorted(list(uposes))
    xpos_tag_list = sorted(list(xposes))
    print(f"\nWe have {len(upos_tag_list)} UPOS tags and {len(xpos_tag_list)} XPOS tags.\n")

    my_collator = MyCollator(tokenizer=tokenizer, upos_tag_list=upos_tag_list, xpos_tag_list=xpos_tag_list)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True,
                                  collate_fn=my_collator, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False,
                                collate_fn=my_collator, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False,
                                 collate_fn=my_collator, pin_memory=True)


    itt = 0

    valid_data = {
        "loss": [],
        "upos_f1": [],
        "upos_precision": [],
        "upos_recall": [],
        "xpos_f1": [],
        "xpos_precision": [],
        "xpos_recall": []
    }
    test_data = {
        "loss": [],
        "upos_f1": [],
        "upos_precision": [],
        "upos_recall": [],
        "xpos_f1": [],
        "xpos_precision": [],
        "xpos_recall": []
    }

    while itt < experiment_iterations:
        print("Running experiment {}/{}".format(itt + 1, experiment_iterations))

        model = TransformerModel(
            model_name=automodel_name,
            upos_tag_list=upos_tag_list,
            xpos_tag_list=xpos_tag_list,
            lr=lr,
            model_max_length=model_max_length
        )

        early_stop = EarlyStopping(
            monitor='valid/join_accuracy',
            patience=5,
            verbose=True,
            mode='max'
        )

        trainer = pl.Trainer(
            gpus=gpus,
            callbacks=[early_stop],
            limit_train_batches=24,
            limit_val_batches=3,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=1.0,
            checkpoint_callback=False
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        result_valid = trainer.test(model, val_dataloader)[0]
        result_test = trainer.test(model, test_dataloader)[0]

        with open("results_pos_{}_of_{}.json".format(itt + 1, args.experiment_iterations), "w") as f:
            json.dump(result_test, f, indent=4, sort_keys=True)

        valid_data["loss"].append(result_valid['test/loss'])
        valid_data["upos_f1"].append(result_valid['test/upos_f1'])
        valid_data["upos_precision"].append(result_valid['test/upos_precision'])
        valid_data["upos_recall"].append(result_valid['test/upos_recall'])
        valid_data["xpos_f1"].append(result_valid['test/xpos_f1'])
        valid_data["xpos_precision"].append(result_valid['test/xpos_precision'])
        valid_data["xpos_recall"].append(result_valid['test/xpos_recall'])

        test_data["loss"].append(result_test['test/loss'])
        test_data["upos_f1"].append(result_test['test/upos_f1'])
        test_data["upos_precision"].append(result_test['test/upos_precision'])
        test_data["upos_recall"].append(result_test['test/upos_recall'])
        test_data["xpos_f1"].append(result_test['test/xpos_f1'])
        test_data["xpos_precision"].append(result_test['test/xpos_precision'])
        test_data["xpos_recall"].append(result_test['test/xpos_recall'])

        itt += 1

    if save_model is True:
      print("\nSaving model to trained_model/")
      model.model.save_pretrained("trained_model/")
      model.tokenizer.save_pretrained("trained_model/")

    print("\nDone, writing results...")

    result = {
        "valid_loss": sum(valid_data["loss"]) / args.experiment_iterations,
        "valid_upos_f1": sum(valid_data["upos_f1"]) / args.experiment_iterations,
        "valid_upos_precision": sum(valid_data["upos_precision"]) / args.experiment_iterations,
        "valid_upos_recall": sum(valid_data["upos_recall"]) / args.experiment_iterations,
        "valid_xpos_f1": sum(valid_data["xpos_f1"]) / args.experiment_iterations,
        "valid_xpos_precision": sum(valid_data["xpos_precision"]) / args.experiment_iterations,
        "valid_xpos_recall": sum(valid_data["xpos_recall"]) / args.experiment_iterations,
        "test_loss": sum(test_data["loss"]) / args.experiment_iterations,
        "test_upos_f1": sum(test_data["upos_f1"]) / args.experiment_iterations,
        "test_upos_precision": sum(test_data["upos_precision"]) / args.experiment_iterations,
        "test_upos_recall": sum(test_data["upos_recall"]) / args.experiment_iterations,
        "test_xpos_f1": sum(test_data["xpos_f1"]) / args.experiment_iterations,
        "test_xpos_precision": sum(test_data["xpos_precision"]) / args.experiment_iterations,
        "test_xpos_recall": sum(test_data["xpos_recall"]) / args.experiment_iterations
    }

    with open("results_pos_of_{}.json".format(args.model_name.replace("/", "_")), "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)

    from pprint import pprint
    pprint(result)

def download_data():
    import requests

    if not os.path.exists('ro_rrt-ud-train.conllu'):
        r = requests.get("https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master/ro_rrt-ud-train.conllu", allow_redirects=True)
        open('ro_rrt-ud-train.conllu', 'wb').write(r.content)
    if not os.path.exists('ro_rrt-ud-dev.conllu'):
        r = requests.get("https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master/ro_rrt-ud-dev.conllu", allow_redirects=True)
        open('ro_rrt-ud-dev.conllu', 'wb').write(r.content)
    if not os.path.exists('ro_rrt-ud-test.conllu'):
        r = requests.get("https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master/ro_rrt-ud-test.conllu", allow_redirects=True)
        open('ro_rrt-ud-test.conllu', 'wb').write(r.content)

if __name__ == "__main__":
    download_data()

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_name', type=str,
                        default="dumitrescustefan/bert-base-romanian-cased-v1")  # xlm-roberta-base
    parser.add_argument('--tokenizer_name', type=str, default=None)
    parser.add_argument("--train_file", type=str, default="ro_rrt-ud-train.conllu")
    parser.add_argument("--validation_file", type=str, default="ro_rrt-ud-dev.conllu")
    parser.add_argument("--test_file", type=str, default="ro_rrt-ud-test.conllu")
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--experiment_iterations', type=int, default=1)
    parser.add_argument('--results_file', type=str, default=None)
    parser.add_argument('--save_model', type=str, default=False)

    args = parser.parse_args()

    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name

    run_evaluation(
        automodel_name = args.model_name,
        tokenizer_name = args.tokenizer_name,
        model_max_length = 512,
        train_file = args.train_file,
        validation_file = args.validation_file,
        test_file = args.test_file,
        gpus = args.gpus,
        batch_size = args.batch_size,
        accumulate_grad_batches = args.accumulate_grad_batches,
        lr = args.lr,
        experiment_iterations = args.experiment_iterations,
        results_file = args.results_file,
        save_model=args.save_model
    )
