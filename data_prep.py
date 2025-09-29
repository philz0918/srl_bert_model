from typing import List, Dict, Optional
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score
import json



#Create data instance, words: tokenized word list, predicte_word_idx: index for predicte, labels: Semantic roles
!@dataclass
class SRLSample():
    def __init__(self, words: List[str], predicate_word_idx: int, labels: List[str], predicate_form: Optional[str] = None):
        self.words = words
        self.predicate_word_idx = predicate_word_idx
        self.labels = labels
        self.predicate_form = predicate_form


#To Leah: SRL Sample is object for each dataset so we need another code for each instance(words, predicate_word_idx, labels) into list of SRLSample objects

def create_srl_samples(data_path):
  samples = []
  with open(data_path, 'r', encoding='utf-8') as f: 
    for line in f: 
      data = json.loads(line)
      samples.append(SRLSample(**data))

  return samples


#Example

#if __name__ == '__main__'

# data_class_train = create_srl_samples('/content/drive/MyDrive/Dissertation/srl_synthetic_100.jsonl')

# data_class_dev = create_srl_samples('/content/drive/MyDrive/Dissertation/srl_synthetic_dev_10.jsonl')

# data_class_test = create_srl_samples('/content/drive/MyDrive/Dissertation/srl_synthetic_test_10.jsonl')


class SRLDataset(Dataset):
    """
    Expects samples at WORD-level. We build BERT inputs as:
      [CLS] <sentence (wordpiece)> [SEP] <predicate (wordpiece)> [SEP]
    We keep:
      - wordpiece indices for each word's FIRST subtoken (to pool BERT to word level)
      - sentence lengths
      - predicate's WORD index within the sentence (for gp from BiLSTM outputs)
    """
    def __init__(self, samples: List[SRLSample], tokenizer: AutoTokenizer, label2id: Dict[str, int], max_length: int = 256, debug_print= False):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.max_length = max_length
        self.debug_print = debug_print

    def __len__(self):
        return len(self.samples)

    def _tokenize_sentence(self, words: List[str]):
        # Tokenize sentence as split words to preserve word boundaries
        enc_sent = self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )
        return enc_sent  # dict with 'input_ids'

    def _tokenize_predicate(self, form: str):
        enc_pred = self.tokenizer(
            form,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )
        return enc_pred

    def __getitem__(self, idx):

        instance = self.samples[idx]
        words = instance.words
        n_words = len(words)
        assert 0 <= instance.predicate_word_idx < n_words, "Bad predicate index."

        pred_form = instance.predicate_form if instance.predicate_form is not None else words[instance.predicate_word_idx]

        # Tokenize sentence and predicate separately (Text -> numeric value)
        enc_sent = self._tokenize_sentence(words)
        enc_pred = self._tokenize_predicate(pred_form)

        # print("This is enc_sent {}, this is enc_prec {} \n".format(enc_sent, enc_pred))
      

        sent_wp_ids = enc_sent["input_ids"]                       # list[int]
        pred_wp_ids = enc_pred["input_ids"]                       # list[int]

        # Build final input ids and token type ids Here we added SEP for predicates create new input ids
        # segment A (0): [CLS] sentence [SEP]
        # segment B (1): predicate [SEP]
        # [CLS] sentence [SEP] predicte [SEP]
        # [CLS] sentence [SEP] ARG0_token [SEP] ARG1_token [SEP] ARG2_token [SEP] -> Model for emotion, formality and politeness
        input_ids = [self.tokenizer.cls_token_id] + sent_wp_ids + [self.tokenizer.sep_token_id] \
                    + pred_wp_ids + [self.tokenizer.sep_token_id]

        # token_type_ids: 0 for [CLS] + sentence + [SEP], 1 for predicate + [SEP]
        ttids = [0] * (1 + len(sent_wp_ids) + 1) + [1] * (len(pred_wp_ids) + 1)

        # Build mapping: each WORD -> index of its FIRST wordpiece inside the FULL sequence
        # We iterate tokenizer.word_ids() by re-tokenizing with special tokens for alignment
        # Simpler: reconstruct with pre-known structure:
        # [CLS] at 0; sentence starts at 1; we need mapping from word index to its FIRST wordpiece offset in `sent_wp_ids`.
        # We'll align by re-tokenizing sentence with is_split_into_words and reading the mapping.
        # HuggingFace trick: get word_ids requires encoding with add_special_tokens=True, so let's do that quickly:
        tmp = self.tokenizer(words, is_split_into_words=True, return_offsets_mapping=False)
        word_ids = tmp.word_ids()
        # print("This is tmp {}, word_ids {}\n".format(tmp, word_ids))
        # Now, tmp.input_ids == [CLS] + sent_wp + [SEP]; positions:
        #   0: CLS, 1..1+len(sent_wp_ids)-1: sentence, 1+len(sent_wp_ids): SEP
        # We need FIRST position per word_id in this tmp encoding.
        first_wp_pos_in_full = []
        seen = set()
        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid not in seen:
                seen.add(wid)
                first_wp_pos_in_full.append(pos)  # pos in tmp sequence
        # Sort by wid order to align [0..n_words-1]
        # word_ids may produce first_wp_pos_in_full in increasing pos order, but ensure length correctness:
        # print("This is first_wp_posin_full {}\n".format(first_wp_pos_in_full))
        first_wp_pos_in_full_sorted = [None] * n_words
        # Build first index per wid:
        first_pos_by_wid = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in first_pos_by_wid:
                first_pos_by_wid[wid] = pos
        for wid in range(n_words):
            first_wp_pos_in_full_sorted[wid] = first_pos_by_wid[wid]

        #first_wp_pos_in_full_sorted is the indices without special tokens (e.g., CLS, SEP)

        # Convert those positions (which refer to tmp with specials) to positions in our final input (with extra predicate segment).
        # In tmp: [CLS] sentence_wp [SEP]
        # In final: [CLS] sentence_wp [SEP] predicate_wp [SEP]
        # So for any position 'pos' inside tmp, it points to the SAME index in final, since the prefix is identical up to first [SEP].
        word_first_wp_fullidx = first_wp_pos_in_full_sorted  # list[int] length = n_words

        # Labels to IDs
        label_ids = [self.label2id[lbl] for lbl in instance.labels]
        assert len(label_ids) == n_words

        # Predicate indicator at word level (0/1)
        indicator = [0] * n_words
        indicator[instance.predicate_word_idx] = 1

        # [0,0,0,0,0] -> [0,0,1,0,0]

        # Attention mask for the full input
        attention_mask = [1] * len(input_ids)

        # Truncate if needed (rare for normal SRL sentences but keep safe)
        if len(input_ids) > self.max_length:
            # We only truncate the predicate side if absolutely necessary; for simplicity, just clip tail.
            input_ids = input_ids[:self.max_length]
            ttids = ttids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            # NOTE: word_first_wp_fullidx could reference beyond max_length in pathological cases.
            max_pos = self.max_length - 1
            word_first_wp_fullidx = [min(p, max_pos) for p in word_first_wp_fullidx]

        if self.debug_print:
            toks_debug = self.tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
            print("[DeBug idx = {}]".format(idx)+" ".join(toks_debug))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(ttids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "word_first_wp_fullidx": torch.tensor(word_first_wp_fullidx, dtype=torch.long),  # [n_words]
            "labels": torch.tensor(label_ids, dtype=torch.long),                               # [n_words]
            "indicator": torch.tensor(indicator, dtype=torch.long),                            # [n_words]
            "sent_len": torch.tensor(len(words), dtype=torch.long),
            "pred_word_idx": torch.tensor(instance.predicate_word_idx, dtype=torch.long)
        }


def srl_collate(batch: List[Dict], pad_token_id: int, pad_label_id: int = -100):
    """
    Pads full BERT inputs to same length; also pads word-level tensors to max sentence length.
    Returns tensors ready for the model.
    """
    B = len(batch)
    # Full sequence padding
    max_L = max(item["input_ids"].size(0) for item in batch)
    # print("This is B {}, max_L {}".format(B,max_L))
    #make tensor B rows and Max_L columns
    input_ids = torch.full((B, max_L), pad_token_id, dtype=torch.long)
    token_type_ids = torch.zeros((B, max_L), dtype=torch.long)
    attention_mask = torch.zeros((B, max_L), dtype=torch.long)

    # Word-level padding
    max_n = max(int(item["sent_len"]) for item in batch)
    word_first_wp_fullidx = torch.full((B, max_n), -1, dtype=torch.long)
    labels = torch.full((B, max_n), pad_label_id, dtype=torch.long)
    indicator = torch.zeros((B, max_n), dtype=torch.long)
    sent_lens = torch.zeros((B,), dtype=torch.long)
    pred_word_idx = torch.zeros((B,), dtype=torch.long)
    sentence_mask = torch.zeros((B, max_n), dtype=torch.bool)

    for i, item in enumerate(batch):
        # print("This is item {}".format(item))
        L = item["input_ids"].size(0)
        input_ids[i, :L] = item["input_ids"]
        token_type_ids[i, :L] = item["token_type_ids"]
        attention_mask[i, :L] = item["attention_mask"]

        n = int(item["sent_len"])
        word_first_wp_fullidx[i, :n] = item["word_first_wp_fullidx"]
        labels[i, :n] = item["labels"]
        indicator[i, :n] = item["indicator"]
        sent_lens[i] = n
        pred_word_idx[i] = item["pred_word_idx"]
        sentence_mask[i, :n] = True

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "word_first_wp_fullidx": word_first_wp_fullidx,  # [B, max_n] (full-seq positions; -1 for pad)
        "sentence_mask": sentence_mask,                  # [B, max_n] (bool mask for valid words)
        "labels": labels,                                # [B, max_n] (pad_label_id for pad)
        "indicator": indicator,                          # [B, max_n] 0/1
        "sent_lens": sent_lens,                          # [B]
        "pred_word_idx": pred_word_idx                   # [B]
    }


def data_processing_for_loader(train_dev_test: List[SRLSample], train_sample: List[SRLSample], dev_sample: List[SRLSample], test_sample: List[SRLSample], tokenizer):

  '''
  train_dev_test is an appended list of Train/Dev/Test SRLSamples
  train_sample is a list of SRLSample
  dev_sample is a list of SRLSample
  test_sample is a list of SRLSample
  '''

  label2id = {}
  for s in train_dev_test:
    for l in s.labels:
        label2id.setdefault(l, len(label2id))
  id2label = {v: k for k, v in label2id.items()}

  #train before loader
  train_bf_loader = SRLDataset(train_sample, tokenizer, label2id, max_length = 128, debug_print = False)
  dev_bf_loader = SRLDataset(dev_sample, tokenizer, label2id, max_length = 128, debug_print = False)
  test_bf_loader = SRLDataset(test_sample, tokenizer, label2id, max_length = 128, debug_print = False)

  return train_bf_loader, dev_bf_loader, test_bf_loader, label2id, id2label
