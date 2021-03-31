import os, sys
import random
import pickle
import typing as T

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

PROJ_PATH = "."
DATA_PATH = os.path.join(PROJ_PATH, "data")
FINAL_MODEL_PATH = os.path.join(PROJ_PATH, "model")

TEST_DATA_PATH = os.path.join(PROJ_PATH, "data", "task1_test_for_user.parquet")
LABEL_ENCODER_PATH = os.path.join(PROJ_PATH, "label_encoder.pickle")

SEED = 42
TOKENIZE_BATCH_SIZE = 256
BATCH_SIZE = 64


class CustomBertForSequenceClassification(BertPreTrainedModel):
    """
    With weighted average of hidden states and multisample dropout
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_layers = outputs[2]

        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2
        )
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)

        # multisample dropout (wut): https://arxiv.org/abs/1905.09788
        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(cls_output)) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss, logits=logits)


def set_seed(seed_val: int = 42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def load_pickle(filepath):
    with open(filepath, "rb") as handle:
        b = pickle.load(handle)
    return b


class TokenizeFunction:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, examples):
        return self._tokenizer(
            examples,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
            # return_special_tokens_mask=True,
        )


def get_test_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def get_dataset(
    tokenizer, texts: np.ndarray, tokenize_batch_size: int = 128
) -> TensorDataset:
    tokenize_function = TokenizeFunction(tokenizer)
    pbar = range(0, len(texts), tokenize_batch_size)

    input_ids = []
    attention_mask = []
    for i in pbar:
        batch = list(texts[i : i + tokenize_batch_size])
        out_batch = tokenize_function(batch)
        input_ids.append(out_batch["input_ids"])
        attention_mask.append(out_batch["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    dataset = TensorDataset(input_ids, attention_mask)

    return dataset


def predict(model, dataloader: DataLoader) -> np.ndarray:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    outputs = []
    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        with torch.no_grad():
            logits = model(b_input_ids, attention_mask=b_attention_mask)[0]
        probs = F.softmax(logits, dim=1)
        outputs.append(probs)

    predictions = torch.cat(outputs).cpu().numpy()
    predictions = predictions.argmax(axis=1)
    return predictions


def main():
    set_seed(seed_val=SEED)

    # get data
    df = get_test_data(path=TEST_DATA_PATH)
    label_encoder = load_pickle(LABEL_ENCODER_PATH)

    # get models
    tokenizer = AutoTokenizer.from_pretrained(
        FINAL_MODEL_PATH, do_lower_case=True, use_fast=True
    )
    # model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL_PATH)
    model = CustomBertForSequenceClassification.from_pretrained(
        FINAL_MODEL_PATH, output_hidden_states=True
    )

    # get dataloader
    dataset = get_dataset(
        tokenizer,
        texts=np.array(df["item_name"]),
        tokenize_batch_size=TOKENIZE_BATCH_SIZE,
    )
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # get predictions
    predictions = predict(model=model, dataloader=dataloader)
    answers = label_encoder.inverse_transform(predictions)

    # save
    df["pred"] = answers
    df[["id", "pred"]].to_csv("answers.csv", index=None)


if __name__ == "__main__":
    main()
