# https://www.youtube.com/watch?v=U0s0f995w14&t=1755s
import torch 
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embedding_size : int , num_heads : int)
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_Size
        self.num_heads = heads
        self.head_dim = embedding_size // heads

        assert (self.head_dim * heads == embedding_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dum, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dum, bias = False)

        self.fc_out = nn.Linear(self.embedding_size, self.embedding_size)
    
    def forward(self, values, keys, query, mask):
        # Number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        # Einsum notation here is great
        # n -> batch size
        # q -> query length
        # h -> num head
        # d -> head dimension
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # softmax(QK/sqrt(d))
        # Can normalize sentence scores in source sentence or destination sentence to figure out what you're paying attention to
        attention = torch.softmax(energy / (self.embedding_size ** (1/2)), dim=3)

        # Multiply by value
        out = torch.einsum("nhql,nlhd->nqhd",[attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )

        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embedding_size : int, heads : int, dropout : float, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, heads)
        
        # Layer norm takes an average for every single example instead of per batch
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        # Project to higher dimension and back to original
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion*embedding_size)
            nn.ReLu(),
            nn.Linear(forward_expansion*embedding_size, embedding_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        # attention + query is the skip connection
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embedding_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length):

        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.device = device

        # TODO: Explain nn.Embedding
        self.word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embeddding_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)



    def forward(self, x, mask):
        N, seq_length = x.shape

        # TODO: https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
        # Each row of x - i.e each sentence gets its own position vector duplicated
        # seq_length means shorter sentences are padded
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            # In encoding layer all the inputs are the same
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, heads)
        self.norm = nn.LayerNorm(embedding_size)
        self.transformer_block = TransformerBlock(
            embedding_size, heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)

        query = self.dropout(self.norm(attention + x))

        out = self.transformer_block(value, key, query, src_mask)
        
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embedding_size,
        num_layers, heads,
        forward_expansion,
        dropout,
        device,
        max_length):

        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embedding_size, heads, forward_epxansion, dropout, device)
            for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc_out(x)
        return out

class Tranformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embedding_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embedding_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # (N, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)

        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)

        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out


# if __name__ == "__main__":


