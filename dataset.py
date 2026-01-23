import torch 
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len) -> None:

        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src #Tokenizer of source Language
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        #creating special Tokens for start and End of sentences  use Method token_to_id 
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id("[SOS]")],dtype = torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id("[EOS]")],dtype = torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id("[PAD]")],dtype = torch.int64)

    def __len__(self):

        return len(self.ds)

    #using __getitem__ to fetch one single sentence from the opus dataset 
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        #Tokenize the text and map to Token id which are then looked up by the src_embed,tgt_embed method in MODEL.PY
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        #input has fixed sequence length,so we do padding
        # We need seq_len - number of tokens - 2 (for SOS and EOS)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens <0:
            raise ValueError('sentence chota kar ')


        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens,dtype=torch.int64 )

            ]
        )
        #Only sos in the decoder  input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens,dtype= torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens,dtype=torch.int64)#paddings 

            ]
        )
          #Add only eos in the Label  
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens,dtype= torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens,dtype=torch.int64)



            ]

        )
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len


        return {
            "encoder_input":encoder_input, #(size = (seq_len))
            "decoder_input": decoder_input,#(size = (seq_len))
            #encoder mask so that the padding do not take part in calculating attention score
            "encoder_mask" : (encoder_input !=self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            #causal Mask so that the decoder cannot see the future Words
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0) &causal_mask(decoder_input.size(0)),
            "label":label,
            "src_text": src_text,
            "tgt_text" :tgt_text




            
        }


def causal_mask(size):
    #mask the element under the diagonal in the matrix so Decoder csnnot see the future words


    mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int) #diagonal =1 does not include diagonal in the 
    return mask == 0 # all lower values are set as true
