import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoConfig

class PredicateAwareSRL(nn.Module):
    def __init__(self,
                 bert_name: str,
                 num_labels: int,
                 use_indicator: bool = True,
                 indicator_dim: int = 10,          # CHANGED: 10-dim predicate indicator
                 lstm_hidden: int = 768,           # CHANGED: BiLSTM hidden size = 768 (bidirectional)
                 mlp_hidden: int = 300,            # CHANGED: MLP hidden size = 300
                 dropout: float = 0.1,
                 use_distance: bool = True,        # NEW: enable relative position (distance) embeddings
                 pos_dim: int = 50,                # NEW: size of position embedding (random init, trainable)
                 max_distance: int = 128):         # NEW: clamp |i - p| to this range for bucketing
        super().__init__()
        self.config = AutoConfig.from_pretrained(bert_name)
        self.bert = AutoModel.from_pretrained(bert_name)
        self.use_indicator = use_indicator

        # --- Input dim to BiLSTM = BERT_dim + (indicator_dim) + (pos_dim)
        bert_dim = self.config.hidden_size
        in_dim = bert_dim + (indicator_dim if use_indicator else 0)

        # Two rows which indicate 0 not predicate 1 is predicate, so need to 2 embedding (rows)
        # num_embeddings (int) – size of the dictionary of embeddings
        # embedding_dim (int) – the size of each embedding vector

        if use_indicator:
            self.indicator_emb = nn.Embedding(2, indicator_dim)  # 0/1 → 10-dim (random init, trainable)  # CHANGED

        self.use_distance = use_distance                         # NEW
        self.max_distance = max_distance                          # NEW
        if use_distance:
            # Distance buckets: [-max_distance .. +max_distance] → indices [0 .. 2*max_distance]
            self.pos_emb = nn.Embedding(2 * max_distance + 1, pos_dim)  # NEW (random init, trainable)
            in_dim += pos_dim                                         # NEW

        # BiLSTM (bidirectional): total output dim = lstm_hidden
        self.bilstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden // 2,  # bi → half per direction
            num_layers=1,
            dropout=0.0,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        # Classifier: concat(g_i, gp) so input dim = 2 * lstm_hidden
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, mlp_hidden),   # CHANGED (mlp_hidden=300)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_labels)
        )

        self.pad_label_id = -100

    def forward(self,
                input_ids: torch.Tensor,           # [B, L]
                token_type_ids: torch.Tensor,      # [B, L]
                attention_mask: torch.Tensor,      # [B, L]
                word_first_wp_fullidx: torch.Tensor,  # [B, max_n] (positions in full seq; -1 for pad)
                sentence_mask: torch.Tensor,       # [B, max_n] (bool)
                sent_lens: torch.Tensor,           # [B]
                pred_word_idx: torch.Tensor,       # [B]
                indicator: torch.Tensor | None = None,  # [B, max_n] 0/1
                labels: torch.Tensor | None = None):    # [B, max_n]

        B, L = input_ids.size()
        device = input_ids.device

        # ---- BERT encoder
        bert_out = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        H = bert_out.last_hidden_state  # [B, L, D]

        # ---- Subword → word pooling (first subword)

        # Gather sentence word-level representations by taking FIRST subtoken hidden per word
        # Prepare indices (replace -1 with 0 to avoid gather OOB; we'll mask later)
        # This process is required to feed word level to predict BIO and role per word
        #.clone is for deep copy won't change original data

        gather_idx = word_first_wp_fullidx.clone()
        gather_idx[gather_idx < 0] = 0
        gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, H.size(-1))  # [B, max_n, D]
        H_words = torch.gather(H, dim=1, index=gather_idx)                 # [B, max_n, D]
        H_words = H_words * sentence_mask.unsqueeze(-1)                    # zero out pads

        # ---- Concatenate predicate indicator (0/1 → emb)
        # word_first_wp_fullidx: [1, 2, 3, -1, -1]
        # gather_idx after clamp: [1, 2, 3, 0, 0]   # 0 points to [CLS], just a placeholder
        # H_words = gather(H, ...)                  # grabs vectors at positions 1,2,3,0,0
        # sentence_mask:        [1, 1, 1, 0, 0]
        # H_words *= mask →     [vec1, vec2, vec3, 0, 0]   # padded slots zeroed out


        X = H_words
        if self.use_indicator and indicator is not None:
            ind_emb = self.indicator_emb(indicator.clamp(0, 1))            # [B, max_n, 10]  # CHANGED
            X = torch.cat([X, ind_emb], dim=-1)

        # ---- NEW: Relative position (distance-to-predicate) embeddings
        if self.use_distance:
            # positions: 0..max_n-1 per sentence
            max_n = X.size(1)
            positions = torch.arange(max_n, device=device).unsqueeze(0).expand(B, -1)  # [B, max_n]
            rel = positions - pred_word_idx.unsqueeze(1)                               # [B, max_n], can be <0
            rel = rel.clamp(-self.max_distance, self.max_distance) + self.max_distance # shift to [0 .. 2*max_distance]
            pos_feats = self.pos_emb(rel)                                              # [B, max_n, pos_dim]  # NEW
            X = torch.cat([X, pos_feats], dim=-1)                                      # [B, max_n, in_dim]  # NEW

        # ---- BiLSTM (packed)
        lengths = sent_lens.detach().cpu()
        packed = pack_padded_sequence(X, lengths=lengths, batch_first=True, enforce_sorted=False)
        G_packed, _ = self.bilstm(packed)
        G, _ = pad_packed_sequence(G_packed, batch_first=True)      # [B, max_n, lstm_hidden]
        G = self.dropout(G)

        # ---- Predicate hidden (word-level) and concat to every position
        batch_idx = torch.arange(B, device=device)
        gp = G[batch_idx, pred_word_idx.clamp(min=0), :]            # [B, lstm_hidden]
        gp_expanded = gp.unsqueeze(1).expand(-1, G.size(1), -1)     # [B, max_n, lstm_hidden]

        logits = self.classifier(torch.cat([G, gp_expanded], dim=-1))  # [B, max_n, num_labels]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_label_id)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss
