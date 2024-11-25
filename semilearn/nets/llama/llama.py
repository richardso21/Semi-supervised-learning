# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
from peft.utils.peft_types import TaskType
from transformers import LlamaModel


class ClassificationLlama(nn.Module):
    def __init__(self, name, num_classes=2, lora_config=None):
        super(ClassificationLlama, self).__init__()
        # Load pre-trained llama model
        self.llama = LlamaModel.from_pretrained(name)

        # If lora_config is defined, wrap model with peft
        if lora_config is not None:
            self.llama = get_peft_model(self.llama, lora_config)
            self.llama.print_trainable_parameters()

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

        out_dict = self.llama(**x, output_hidden_states=True, return_dict=True)
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
        out_dict = self.llama(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict["last_hidden_state"]
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=""):
        matcher = dict(
            stem=r"^{}bert.embeddings".format(prefix),
            blocks=r"^{}bert.encoder.layer.(\d+)".format(prefix),
        )
        return matcher

    def no_weight_decay(self):
        return []


def llama_8b(pretrained=True, pretrained_path=None, **kwargs):
    model = ClassificationLlama(name="meta-llama/Meta-Llama-3-8B", **kwargs)
    return model


def llama_8b_lora(pretrained=True, pretrained_path=None, **kwargs):
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = ClassificationLlama(
        name="meta-llama/Meta-Llama-3-8B",
        lora_config=lora_config,
        **kwargs,
    )
    return model
