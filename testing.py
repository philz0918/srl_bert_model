from SRL_model import SRL_BERT_model
from collections import Counter
import torch

def bio_to_spans(tags):
    spans = []
    i = 0
    while i < len(tags):
        t = tags[i]
        if t == "O" or t.endswith("-V"):
            i += 1; continue
        if t.startswith("B-"):
            role = t[2:]; j = i + 1
            while j < len(tags) and tags[j] == f"I-{role}":
                j += 1
            spans.append((role, i, j-1))
            i = j
        else:
            i += 1
    return spans

@torch.no_grad()
def eval_span_f1(model, dataloader, id2label, device="cuda"):
    model.eval()
    tp = fp = fn = 0
    for batch in dataloader:
        gold = batch["labels"]               # [B, Lw]
        mask = (gold != -100)

        batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
        logits, _ = model(**batch)
        pred = logits.argmax(-1).cpu()       # [B, Lw]
        print(pred)
        for g_seq, p_seq, m in zip(gold, pred, mask):
            gl = [id2label[int(i)] for i in g_seq[m].tolist()]
            pl = [id2label[int(i)] for i in p_seq[m].tolist()]
            G = Counter(bio_to_spans(gl))
            P = Counter(bio_to_spans(pl))
            # micro counts
            common = G & P
            tp += sum(common.values())
            fp += sum(P.values()) - sum(common.values())
            fn += sum(G.values()) - sum(common.values())

    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, f1


if __name__ =="__main__": 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "/blue/bonniejdorr/youms/SRL-Aware_Model/model/best_srl_Sep_29.ckpt"   # <-- change if needed
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt["hparams"]

    model = SRL_BERT_model.PredicateAwareSRL(**hp).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    label2id = ckpt["label2id"]
    id2label = {v: k for k, v in label2id.items()}

    h = ckpt.get("hparams", {
        "bert_name": "bert-base-cased",
        "num_labels": len(label2id),
        "use_indicator": True,
        "use_distance": True,
        "indicator_dim": 10,
        "lstm_hidden": 768,
        "mlp_hidden": 300,
        "pos_dim": 50,
        "max_distance": 128,
        "dropout": 0.1,
    })

    #test_loader from SRL_BERT_model
    prec, rec, span_f1 = eval_span_f1(model, test_loader, id2label, device=device)
    print(f"[TEST-SPAN] P={prec:.3f} R={rec:.3f} F1={span_f1:.3f}")
