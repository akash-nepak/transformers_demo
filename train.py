import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer  # using Huggingface world_level Tokenizer 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer  #create vocabulary given List of sentence 
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_or_build_tokenizer(config,ds,lang)
    #config of our model, ds = Dataset,lang =,language for which we are building The Tokenizer 
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
