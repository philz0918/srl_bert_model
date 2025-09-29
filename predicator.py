## This is testing 

import torch

@torch.no_grad()
def predict_srl_single(model, tokenizer, words, predicate_word_idx, id2label, device="cuda"):
    # tokenize sentence (no specials)
    sent_enc = tokenizer(
        words, is_split_into_words=True, add_special_tokens=False,
        return_attention_mask=False, return_token_type_ids=False
    )
    sent_wp_ids = sent_enc["input_ids"]
    sent_word_ids = sent_enc.word_ids()

    # first-subword position per word in the FULL sequence: [CLS] sent [SEP] pred [SEP]
    first_pos_by_wid = {}
    for pos, wid in enumerate(sent_word_ids):
        if wid is not None and wid not in first_pos_by_wid:
            first_pos_by_wid[wid] = pos + 1  # +1 for [CLS]
    n_words = len(words)
    word_first_wp_fullidx = torch.tensor(
        [first_pos_by_wid[i] for i in range(n_words)], dtype=torch.long
    ).unsqueeze(0)

    # predicate segment = surface form of the predicate word
    pred_enc = tokenizer(
        [words[predicate_word_idx]], is_split_into_words=True, add_special_tokens=False,
        return_attention_mask=False, return_token_type_ids=False
    )
    pred_wp_ids = pred_enc["input_ids"]

    # assemble full input
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    input_ids = [cls_id] + sent_wp_ids + [sep_id] + pred_wp_ids + [sep_id]
    token_type_ids = [0] * (1 + len(sent_wp_ids) + 1) + [1] * (len(pred_wp_ids) + 1)
    attention_mask = [1] * len(input_ids)

    # tensors
    input_ids     = torch.tensor(input_ids).unsqueeze(0).to(device)
    token_type_ids= torch.tensor(token_type_ids).unsqueeze(0).to(device)
    attention_mask= torch.tensor(attention_mask).unsqueeze(0).to(device)

    sent_len      = torch.tensor([n_words], dtype=torch.long).to(device)
    sentence_mask = torch.ones(1, n_words, dtype=torch.bool).to(device)
    pred_word_idx = torch.tensor([predicate_word_idx], dtype=torch.long).to(device)
    indicator     = torch.zeros(1, n_words, dtype=torch.long).to(device)
    indicator[0, predicate_word_idx] = 1
    word_first_wp_fullidx = word_first_wp_fullidx.to(device)

    # forward
    logits, _ = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        word_first_wp_fullidx=word_first_wp_fullidx,
        sentence_mask=sentence_mask,
        sent_lens=sent_len,
        pred_word_idx=pred_word_idx,
        indicator=indicator,
        labels=None
    )

    pred_ids = logits.argmax(-1).squeeze(0).tolist()
    tags = [id2label[i] for i in pred_ids]
    return tags, logits.squeeze(0).cpu()  # [L_word, num_labels]

def bio_to_spans(tags):
    spans = []
    i = 0
    while i < len(tags):
        t = tags[i]
        if t == "O" or t.endswith("-V"):
            i += 1
            continue
        if t.startswith("B-"):
            role = t[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{role}":
                j += 1
            spans.append((role, i, j-1))
            i = j
        else:
            i += 1
    return spans

@torch.no_grad()
def predict_srl_all_predicates(model, tokenizer, sentence, id2label, device="cuda", prob_threshold=0.50):
    words = sentence.split()
    # find the numeric id for "B-V"
    b_v_id = None
    for k, v in id2label.items():
        if v == "B-V":
            b_v_id = k
            break
    if b_v_id is None:
        raise ValueError("Label set has no 'B-V' tag.")

    results = []
    for p in range(len(words)):
        tags, logits = predict_srl_single(model, tokenizer, words, p, id2label, device=device)
        # check predicate decision at position p
        pred_id_at_p = logits.argmax(-1)[p].item()
        keep = (pred_id_at_p == b_v_id)

        # optional confidence gate
        if prob_threshold is not None:
            probs = torch.softmax(logits[p], dim=-1)
            keep = keep and (probs[b_v_id].item() >= prob_threshold)

        if keep:
            spans = bio_to_spans(tags)
            results.append({
                "predicate_index": p,
                "predicate": words[p],
                "tags": tags,
                "spans": spans
            })
    return words, results



# words, preds = predict_srl_all_predicates(model, tokenizer, sentence, id2label, device=device)


def predicator_srl(sentence):
    words, preds = predict_srl_all_predicates(model, tokenizer, sentence, id2label, device=device)

    return words, preds

if __name__ == "__main__":
    sentence = "Hojeong decide to go to the school"
    words, preds = predicator_srl(sentence)
    print(words)
    for r in preds:
        print(f"Predicate: {r['predicate']} (idx {r['predicate_index']})")
        print("Tags:", list(zip(words, r["tags"])))
        print("Spans:", r["spans"])  # (ROLE, start, end) indices over words
        print("-" * 60)



