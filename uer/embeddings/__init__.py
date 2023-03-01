'''
Date: 2022-09-16 10:24:13
LastEditors: Jagger
Description: 
LastEditTime: 2022-12-30 15:12:00
FilePath: /UER-py/uer/embeddings/__init__.py
'''
from uer.embeddings.dual_embedding import DualEmbedding
from uer.embeddings.word_embedding import WordEmbedding
from uer.embeddings.wordpos_embedding import WordPosEmbedding
from uer.embeddings.wordposseg_embedding import WordPosSegEmbedding
from uer.embeddings.wordsinusoidalpos_embedding import WordSinusoidalposEmbedding
from uer.embeddings.kg_embedding import KgEmbedding


str2embedding = {"word": WordEmbedding, "word_pos": WordPosEmbedding, "word_pos_seg": WordPosSegEmbedding,
                 "word_sinusoidalpos": WordSinusoidalposEmbedding, "dual": DualEmbedding,'kg':KgEmbedding}

__all__ = ["WordEmbedding", "WordPosEmbedding", "WordPosSegEmbedding", "WordSinusoidalposEmbedding",
           "DualEmbedding", "str2embedding"]
