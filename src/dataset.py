import numpy as np
from typing import List, Dict, Any, Tuple
import torch
import json
from torch.utils.data import Dataset
from sentence_transformers import (
    SentenceTransformer,
)  # To be changed with our chosen sentence transformer

"""
JsonlDataset class used to format datasets and prepare for training
"""


class JsonlDataset(Dataset):
    def __init__(self, paths, cfg, encoder=None):
        self.data = []
        self.conversations = []
        self.cfg = cfg
        self.max_seq_length = 0
        self.encoder = encoder or SentenceTransformer("all-mpnet-base-v2")
        for path in paths:
            with open(path) as f:
                currConvo = []
                with open(path, "r") as f:
                    text = f.read()
                    decoder = json.JSONDecoder()
                    pos = 0
                    length = len(text)
                while pos < length:
                    # Skip any whitespace/newlines before the next JSON object
                    while pos < length and text[pos].isspace():
                        pos += 1
                    if pos >= length:
                        break

                    # Decode the next JSON object
                    obj, idx = decoder.raw_decode(text[pos:])
                    if "question" not in obj:
                        break
                    else:
                        question = obj["question"]
                        retrievals = obj["retrieval"]

                        # Encode question + doc
                        q_emb = self.encoder.encode(question)
                        d_embs = self.encoder.encode(retrievals[0])
                        # Each entry now has question embedding and doc embeddings
                        currConvo.append((q_emb, d_embs))

                    # Move the position forward
                    pos += idx

            self.conversations.append(currConvo)
        self._build_examples()

    def _build_examples(self):
        """
        For each conversation, build training examples:
          input = [q_0, ..., q_{n-1}] + [d_{n-1}]
          target = q_n
        """
        for conversation in self.conversations:
            convoLength = len(conversation)
            self.max_seq_length = max(convoLength, self.max_seq_length)
            print(f"Max convo length: {self.max_seq_length}")

            for i in range(1, convoLength):
                prev_queries = [conversation[k][0] for k in range(i)]
                prev_doc = conversation[i - 1][1]
                target_query = conversation[i][0]
            example = {"queries": prev_queries, "doc": prev_doc, "target": target_query}
            self.data.append(example)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        prev_queries = example["queries"]  # list of query embeddings
        doc_emb = example["doc"]  # single doc embedding
        target_query = example["target"]  # single query embedding

        seq = prev_queries + [doc_emb]  # [q0, q1, ..., q_{n-1}, d_{n-1}]
        seg_ids = [0] * len(prev_queries) + [1]  # identifies query emb and doc emb

        input_embs = torch.tensor(
            np.array(seq), dtype=torch.float
        )  # dim: (seq_len, emb_dim)
        seg_ids = torch.tensor(seg_ids, dtype=torch.long)
        attn_mask = torch.ones(len(seq), dtype=torch.bool)  # all real, no pad yet
        labels = torch.tensor(target_query, dtype=torch.float)

        return {
            "input_embs": input_embs,  # (seq_len, emb_dim)
            "attn_mask": attn_mask,  # (seq_len,)
            "seg_ids": seg_ids,  # (seq_len,)
            "labels": labels,  # (emb_dim,)
        }


def collate_batch(batch: List[Tuple[np.ndarray, np.ndarray]]):

    max_len = max(item["input_embs"].shape[0] for item in batch)
    emb_dim = batch[0]["input_embs"].shape[1]

    input_embs = []
    attn_masks = []
    seg_ids = []
    labels = []

    for item in batch:
        seq_len = item["input_embs"].shape[0]

        # Pad embeddings
        padded_embs = torch.zeros((max_len, emb_dim), dtype=torch.float)
        padded_embs[:seq_len] = item["input_embs"]

        # Pad masks
        padded_mask = torch.zeros(max_len, dtype=torch.bool)
        padded_mask[:seq_len] = item["attn_mask"]

        # Pad seg_ids
        padded_seg = torch.zeros(max_len, dtype=torch.long)
        padded_seg[:seq_len] = item["seg_ids"]

        input_embs.append(padded_embs)
        attn_masks.append(padded_mask)
        seg_ids.append(padded_seg)
        labels.append(item["labels"])

    return {
        "input_embs": torch.stack(input_embs),  # (batch, max_len, emb_dim)
        "attn_mask": torch.stack(attn_masks),  # (batch, max_len)
        "seg_ids": torch.stack(seg_ids),  # (batch, max_len)
        "labels": torch.stack(labels),  # (batch, emb_dim)
    }
