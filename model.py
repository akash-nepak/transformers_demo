import torch 
import torch.nn as nn
import numpy as np
import math


class InputEmbeddings(nn.Module):

    def __init__(self,d_model: int,vocab_size: int):

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model) # EMBEDDINGS words from the vocab

    
    def forward(self,x): # returns the embeddings of the sequence scaled by sqrt(dmodel)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self,d_model : int,seq_len: int,dropout: float):

        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len,d_model) #creating positional encoding matrix
        
        position = torch.arange(0,seq_len, dtype = torch.float).unsqueeze(1) # creating a vectpr to store the pos of the word
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000)/d_model)) #converting from expo to log for num stability
        
        pe[:, 0::2] = torch.sin(position * div_term )
        pe[:,1::2] = torch.cos(position * div_term)
 

        pe = pe.unsqueeze(0) #shepe of the matrix (1,seq_lenghth, d_model)

        self.register_buffer('pe',pe)  #saving the state along with current state in a register buffer

    
    def forward(self,x): #x shape (batch, seq_lenghth,dmodel)

        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)  # some dimensions are disabled so model does not overfit
    



class LayerNormalization(nn.Module): # mean and var are caluclated for each token embeddings seperatly 
    
    def __init__(self,eps : float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  #making the parameter learnable
        self.bias =  nn.Parameter(torch.ones(1)) # added params

    def forward(self,x):

     mean = x.mean(dim = -1,keepdim = True)
     std = torch.sqrt(x.var(dim = -1, keepdim = True)+ self.eps)
     return self.aplha * (x-mean)/ (std) + self.bias


class FeedForwardBlock(nn.Modules):

    def __init__(self,d_model : int, d_ff :int,dropout : float) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(d_model,d_ff) #projecting into a higher dimensions 4* d_model
        self.dropout 









    
        







        








    

