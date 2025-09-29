from SRL_MODEL import data_prep, SRL_BERT_model
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import pickle

def save_pkl(tgt_list, svg_path):
    with open(svg_path, "wb") as f:
        pickle.dump(tgt_list, f)

def load_pkl(path) :
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device="cuda",
    scheduler=None,
    grad_accum_steps=1,
    amp=True,
    max_grad_norm=1.0,
):
    model.train()
    total_loss, n_steps = 0.0, 0

    use_amp = amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader, 1):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            _, loss = model(**batch)  # model must return (logits, loss)

        total_loss += float(loss.detach().item())
        n_steps += 1

        loss = loss / grad_accum_steps  # for accumulation

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

    return total_loss / max(1, n_steps)

#This is Validation
@torch.no_grad()
def eval_loss_and_token_f1(model, dataloader, id2label=None, device="cuda", average="micro"):

    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_golds = [], []

    for batch in dataloader:
        gold = batch["labels"]                  # keep on CPU for masking
        mask = (gold != -100)

        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits, loss = model(**batch)           # loss computed once here
        total_loss += float(loss.item()); n_batches += 1

        preds = logits.argmax(-1).cpu()
        all_preds.extend(preds[mask].tolist())
        all_golds.extend(gold[mask].tolist())

    f1 = f1_score(all_golds, all_preds, average=average)
    return total_loss / max(1, n_batches), f1


if __name__ =='__main__':
  bert_name = "bert-base-cased"
  tokenizer = AutoTokenizer.from_pretrained(bert_name)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

  #data_class_train/dev/test from data_prep
  train_dev_test_data = data_class_train + data_class_dev + data_class_test
  train_bf_loader, dev_bf_loader,test_bf_loader, label2id, id2label = data_prep.data_processing_for_loader(train_dev_test_data, data_class_train, data_class_dev, data_class_test, tokenizer)

  pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
  collate = lambda b: data_prep.srl_collate(b, pad_token_id=pad_token_id, pad_label_id=-100)

  train_loader = data_prep.DataLoader(train_bf_loader, batch_size=16, shuffle=True, collate_fn=collate)
  dev_loader = data_prep.DataLoader(dev_bf_loader, batch_size=16, shuffle=False, collate_fn=collate)
  test_loader = data_prep.DataLoader(test_bf_loader, batch_size=16, shuffle=False, collate_fn=collate)

  # bert_name = "bert-base-cased"
  # tokenizer = AutoTokenizer.from_pretrained(bert_name)

  # device = "cuda" if torch.cuda.is_available() else "cpu"

  model = SRL_BERT_model.PredicateAwareSRL(
      bert_name=bert_name,
      num_labels=len(label2id),
      use_indicator=True,
      use_distance =True,
      indicator_dim= 10,
      lstm_hidden=768,
      mlp_hidden=300,
      pos_dim= 50,
      max_distance = 128,
      dropout=0.1
  ).to(device)

  # Optimizer (you may want to use AdamW with weight decay and a scheduler)
  num_epochs = 12
  grad_accum_steps = 1 
  optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

  # # Train a couple of epochs (on toy data this is just to check shapes run)
  # for epoch in range(3):
  #     tr_loss = train_one_epoch(model, train_loader, optimizer, device=device)
  #     f1 = evaluate_token_f1(model, dev_loader, id2label=id2label, device=device)
  #     print(f"Epoch {epoch+1} | loss={tr_loss:.4f} | token-F1={f1:.4f}")

  total_steps = len(train_loader) * num_epochs // max(1, grad_accum_steps)
  warmup_steps = int(0.1 * total_steps)

  scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=warmup_steps,
      num_training_steps=total_steps
  )

  history = {"epoch": [], "train_loss": [], "dev_loss": [], "dev_f1": []}

  best_dev, best_path = -1.0, "best_srl.ckpt"
  for epoch in range(num_epochs):
      tr_loss = train_one_epoch(
          model, train_loader, optimizer, device=device,
          scheduler=scheduler, grad_accum_steps=grad_accum_steps, amp=True, max_grad_norm=1.0
      )
      dev_loss, dev_f1 = eval_loss_and_token_f1(model, dev_loader, id2label, device=device)


      history["epoch"].append(epoch + 1)
      history["train_loss"].append(tr_loss)
      history["dev_loss"].append(dev_loss)
      history["dev_f1"].append(dev_f1)

      print(f"Epoch {epoch+1}: train_loss={tr_loss:.4f}  dev_loss={dev_loss:.4f}  dev_F1={dev_f1:.4f}")

      if dev_f1 > best_dev:
          best_dev = dev_f1
          torch.save({"model_state": model.state_dict(), "label2id": label2id}, best_path)
          print("  ↳ new best dev; saved.")

  save_pkl(history, #save_path_for_loss)

  # best_dev, best_path = -1.0, "best_srl.ckpt"
  # for epoch in range(num_epochs):
  #     tr_loss = train_one_epoch(model, train_loader, optimizer, device=device)
  #     dev_loss, dev_f1 = eval_loss_and_token_f1(model, dev_loader, id2label, device=device)
  #     print(f"Epoch {epoch+1}: train_loss={tr_loss:.4f}  dev_loss={dev_loss:.4f}  dev_F1={dev_f1:.4f}")
  #     if dev_f1 > best_dev:
  #         best_dev = dev_f1
  #         torch.save({"model_state": model.state_dict(), "label2id": label2id}, best_path)
  #         print("  ↳ new best dev; saved.")