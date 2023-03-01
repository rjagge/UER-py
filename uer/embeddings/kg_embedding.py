'''
Date: 2022-12-30 15:04:40
LastEditors: Jagger
Description: 
LastEditTime: 2023-02-23 13:19:33
FilePath: /research/UER-py/uer/embeddings/kg_embedding.py
'''
import torch.nn as nn
import torch

from uer.layers.layer_norm import LayerNorm


class KgEmbedding(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(KgEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        # self.ent_embedding = nn.Embedding(vocab_size, args.emb_size)
        ent_weight = torch.load(args.kg_emb_path)
        self.ent_embedding = nn.Embedding.from_pretrained(ent_weight, freeze=False)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.kg_emb_size)

    def forward(self, src):
        """
        Args:
            src: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        Returns:
            emb: [batch_size x seq_length x hidden_size]
        """
        emb = self.ent_embedding(src)
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb
