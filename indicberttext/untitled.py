import torch
from transformers import AutoTokenizer, AutoModel
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import re

# --- Module-level constants ---
# Using the canonical multilingual model from AI4Bharat for broad language support.
INDIC_BERT_MODEL_NAME = "ai4bharat/indic-bert"
# VITS2's internal hidden dimension. The BERT embeddings will be projected to this size.
# This value should match the `hidden_channels` parameter in your VITS2 config.
VITS2_HIDDEN_DIM = 192

