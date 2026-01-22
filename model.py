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
        
        pe[:, 0::2] = torch.sin(position * div_term ) #APPLYING sin to even pos
        pe[:,1::2] = torch.cos(position * div_term) # applying cos to odd positions 
 

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

     mean = x.mean(dim = -1,keepdim = True) #shape (Batch,mean,1)
     std = torch.sqrt(x.var(dim = -1, keepdim = True)+ self.eps)
     return self.alpha * ((x-mean)/ (std)) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self,d_model : int, d_ff :int,dropout : float) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(d_model,d_ff) #projecting into a higher dimensions 4* d_model
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward (self,x):
        #transforms from (Batch,seq_Len,d_model) --> (BatH, Seq_len,d_ff)

        return self.linear_2(self.dropout(torch.nn.functional.gelu(self.linear_1(x))))

        
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model : int,h: int,dropout:float):

        super().__init__() 

        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h "

        self.d_k = d_model // h 

        self.w_q = nn.Linear(d_model,d_model,)  #creating a learnanble wq,wk,wv matrix
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query,key,value,mask,dropout :nn.Dropout):

        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)#transposing from (B,H,seq,d_model) --> (B,h,Dmod,seq)
        if mask is not None:

            attention_scores.masked_fill_(mask ==0, -1e9)
        attention_scores=attention_scores.softmax(dim = -1) #(Batch, h, S_L,S_l)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value),attention_scores



        




         


    def forward(self,q, k,v,mask): 
        query = self.w_q(q) #(Batch,Seq_Lenghth,D_model) --> (Batch,Seq_Lenghth,D_model)
        key = self.w_k(k)  #(Batch,Seq_Lenghth,D_model) --> (Batch,Seq_Lenghth,D_model)
        value = self.w_v(v) #(Batch,Seq_Lenghth,D_model) --> (Batch,Seq_Lenghth,D_model)
       
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose (1,2).contiguous()

        #he view command reshapes the tensor. It takes the big embedding vector of size d_model at the end and "cuts" it into h pieces, each of size d_k.
        #The .transpose(1, 2) - Arranging for Parallel Processing, and seeing full seequence and focusing different aspects of the same sentence 
        #Transformation: (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_Len, d_k)

        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        #(B,H,Seq,dk) --> (B,seq,h,dk) -->(B,S,D_MODEL)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h *self.d_k) 

        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self,dropout:float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm =LayerNormalization()


    def forward(self,x,sublayer): #here sublayer means a module that should be passed 

        return x + self.dropout(sublayer(self.norm(x))) #pre normalization for more stability


## Building the encoder block ##


class EncoderBlock(nn.Module):

    def __init__(self,self_attention_block : MultiHeadAttentionBlock,feed_forward_block : FeedForwardBlock,dropout : float) ->None:

        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        #creats 2 instances of the ResidualConnections as needed for encoder 

    def forward(self,x,src_mask): # we apply src_mask in encoder to prevent interation of paddings words with input token 

        x = self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x,src_mask) )
        x = self.residual_connections[1](x,self.feed_forward_block) #passes x as first as parameter and then passes feed_forward block as a second parameter as sublayer 


        return x

        
class Encoder(nn.Module): #stacking 'N' encoder layer together

    def __init__(self,layers:nn.ModuleList) -> None: 

        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()


    def forward (self ,x ,mask):  #stacking multiple encoder blocks 

        for layer in self.layers: # output of one encoder is input to other 
            x = layer (x,mask)

        return self.norm(x)
    


class DecoderBlock(nn.Module):

    def __init__(self,self_attention_block :MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock,feed_forward_block : FeedForwardBlock,dropout : float) -> None:

        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout)for _ in range(3)]) # initialising 3 residual blocks for decoder structure 
  
    
    def forward(self,x , encoder_output,src_mask, tgt_mask  ):

        x = self.residual_connection[0](x,lambda x: self.self_attention_block(x,x,x,tgt_mask))
        #cross attention block output of self attention from decoder with keys and values comming from the encoder block
        #only final layer of the encoder layer is fed into cross-attention of  all the layer of Decoder Layer
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connection[2] (x,self.feed_forward_block) 


        return x  
      

class Decoder(nn.Module): # stacking n decoder layers together 

    def __init__(self,layers:nn.ModuleList) -> None:

        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:

            x = layer(x,encoder_output,src_mask,tgt_mask)

        return self.norm(x)
        

class ProjectionLayer(nn.Module): #projecting the seq_length,d_model back to words

    def __init__(self,d_model :int,vocab_size :int) -> None:

        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)


    def forward(self,x): #(Batch,seq_len,d_model) --> (Batch,seq_len,vocab_size)

        return torch.log_softmax(self.proj(x),dim = -1)

        


class Transformer(nn.Module): #building the entire Module

    def __init__(self,encoder :Encoder,decoder :Decoder,src_embed : InputEmbeddings,tgt_embed :InputEmbeddings,src_pos:PositionalEncoding,tgt_pos : PositionalEncoding,projection_layer : ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder =decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer


    def encode(self,src,src_mask):

        src = self.src_embed(src) #
        src = self.src_pos(src) # adding positional encoding 

        return self.encoder(src,src_mask)
   
    def decode(self, encoder_output,src_mask,tgt,tgt_mask ):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    

    def project(self,x):

        return self.projection_layer(x)


def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len :int,d_model :int = 512,N :int = 6,h: int =8,dropout:float = 0.1,d_ff:int = 2048 ) ->Transformer:
    #Defining Hyperparameters for building a  Transformer, like h ,d_ff, d_model
    #N =  no of Encoder Decoder block stacked

    src_embed = InputEmbeddings(d_model,src_vocab_size) # embedding vector for entire vocab
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)

    #Massive matrix of random number ,each row correspons to one unique word in the vocabulary
    #Later, when you pass a sentence like [42, 12, 99] (Token IDs) into src_embed:

    #Lookup: It looks up row 42, row 12, and row 99 in that matrix.
     
    src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model,tgt_seq_len,dropout)

    #creatinge encoder block N arryas stacked together 
    encoder_blocks = []

    for _ in range (N): #create sperate 6 stacked encoder blocks 
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    # creating the decoder block 
    decoder_blocks =[]

    for _ in range (N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)


    encoder = Encoder(nn.ModuleList(encoder_blocks))  #converts into pytorch-native list 
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #projection layer 

    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)


    #creating a Transformer 
    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)


    #initialise the parameters, he's or xavier initialization

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer








