import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader , random_split

from datasets import load_dataset
from tokenizers import Tokenizer  # using Huggingface world_level Tokenizer 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer  #create vocabulary given List of sentence 
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_sentences(ds,lang):

    for item in ds:
        yield item ['translation'][lang] # Extracting a single language from the dataset pair




    pass

def get_or_build_tokenizer(config,ds,lang)
    #config of our model, ds = Dataset,lang =,language for which we are building The Tokenizer 
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) #storing the Tokenizer path'
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) #replacing Unknown Vocab with UNK
        tokenizer.pre_tokenizer = Whitespace() #split by Whitespace
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_senetences(ds,lang),trainer =trainer) #Training the Tokenizer 
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):

    #Downloading dataset from Hugging face
    ds_raw = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}', split = 'train')

    #Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])


    train_ds_size = int (0.9 * len(ds_raw))
    val_ds_size = len(ds_raw ) - train_ds_size
    #spliting ds_raw into train_ds_size and  val_ds_size using random_split method 
    train_ds_raw, val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])



