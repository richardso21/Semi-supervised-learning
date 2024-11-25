# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from transformers import DebertaModel


class ClassificationDeberta(nn.Module):
    def __init__(self, name, num_classes=2):
        super(ClassificationDeberta, self).__init__()
        self.deberta = DebertaModel.from_pretrained(name)
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

        out_dict = self.deberta(**x, output_hidden_states=True, return_dict=True)
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
        out_dict = self.deberta(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict["last_hidden_state"]
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=""):
        matcher = dict(
            stem=r"^{}deberta.embeddings".format(prefix),
            blocks=r"^{}deberta.encoder.layer.(\d+)".format(prefix),
        )
        return matcher

    def no_weight_decay(self):
        return []


def deberta_base(pretrained=True, pretrained_path=None, **kwargs):
    model = ClassificationDeberta(name="microsoft/deberta-base", **kwargs)
    return model


if __name__ == "__main__":
    model = deberta_base()
    print(model)