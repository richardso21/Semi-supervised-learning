# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from transformers import RobertaModel


class ClassificationRoberta(nn.Module):
    def __init__(self, name, num_classes=2):
        super(ClassificationRoberta, self).__init__()
        # Load pre-trained roberta model
        self.roberta = RobertaModel.from_pretrained(name)
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        self.num_features = 768
        self.classifier = nn.Sequential(
            *[nn.Linear(768, 768), nn.GELU(), nn.Linear(768, num_classes)]
        )

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
            return_embed: return word embedding, used for vat
        """
        if only_fc:
            logits = self.classifier(x)
            return logits

        out_dict = self.roberta(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict["last_hidden_state"]
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)

        if only_feat:
            return pooled_output

        logits = self.classifier(pooled_output)
        result_dict = {"logits": logits, "feat": pooled_output}

        if return_embed:
            result_dict["embed"] = out_dict["hidden_states"][0]

        return result_dict

    def extract(self, x):
        out_dict = self.roberta(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict["last_hidden_state"]
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=""):
        matcher = dict(
            stem=r"^{}roberta.embeddings".format(prefix),
            blocks=r"^{}roberta.encoder.layer.(\d+)".format(prefix),
        )
        return matcher

    def no_weight_decay(self):
        return []


def roberta_base(pretrained=True, pretrained_path=None, **kwargs):
    model = ClassificationRoberta(name="FacebookAI/roberta-base", **kwargs)
    return model


def roberta_base_sentiment(pretrained=True, pretrained_path=None, **kwargs):
    model = ClassificationRoberta(
        name="cardiffnlp/twitter-roberta-base-sentiment-latest", **kwargs
    )
    return model


if __name__ == "__main__":
    model = roberta_base()
    print(model)
