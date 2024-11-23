# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .bert import bert_base_cased, bert_base_uncased
from .hubert import hubert_base
from .llama import llama_8b
from .resnet import resnet50
from .roberta import roberta_base
from .t5 import t5_base
from .vit import (vit_base_patch16_96, vit_base_patch16_224,
                  vit_small_patch2_32, vit_small_patch16_224,
                  vit_tiny_patch2_32)
from .wave2vecv2 import wave2vecv2_base
from .wrn import wrn_28_2, wrn_28_8, wrn_var_37_2
from .xlnet import xlnet_base_cased
