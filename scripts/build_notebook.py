"""Generate the tracked Kaggle notebook. Kept as source so cells are easy to audit."""
import json
from pathlib import Path

cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.strip()+"\n"})
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip()+"\n"})

md("""# IntentScope — Fine-Grained Banking Intent Classification with DistilBERT

This notebook fine-tunes DistilBERT on Banking77 with an explicit PyTorch loop. The official test split remains untouched until the best validation checkpoint has been selected. Final metrics are computed independently from stored predictions.

**Offline Kaggle contract:** attach a Kaggle Dataset containing `banking77/` (saved with `DatasetDict.save_to_disk`) and `distilbert-base-uncased/` (saved with `save_pretrained`). A `wheels/` directory is optional. The companion preparation script creates this bundle.""")

md("""## 1–3. Imports, configuration, and reproducibility

Fixed seeds support repeatable experiments. Exact bitwise equality can still vary across GPU, CUDA, and library versions.""")
code(r'''import json, os, random, time
from dataclasses import asdict, dataclass
from pathlib import Path

import datasets, matplotlib.pyplot as plt, numpy as np, pandas as pd, torch, transformers
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()

@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    seed: int = 42
    validation_size: float = 0.15
    max_length: int = 128
    batch_size: int = 64
    epochs: int = 25
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    num_workers: int = 0
    run_baseline: bool = True
    offline: bool = True

CFG = Config()
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
BEST_DIR = OUTPUT_DIR / "best_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("Device:", device)
if device.type != "cuda": print("WARNING: training on CPU will be slow.")''')

md("""## 5. Load Banking77 and create validation data

Validation is used for checkpoint selection; the test split is reserved for one final evaluation. We inspect the schema first and use a stratified split so every intent keeps approximately the same proportion.""")
code(r'''def find_asset_dir(name):
    matches = [p for p in Path("/kaggle/input").rglob(name) if p.is_dir()]
    return matches[0] if matches else None

dataset_asset = find_asset_dir("banking77")
model_asset = find_asset_dir("distilbert-base-uncased")
if dataset_asset is None or model_asset is None:
    raise FileNotFoundError("Attach a Dataset containing banking77/ and distilbert-base-uncased/.")

categories_path = dataset_asset / "categories.json"
train_path = dataset_asset / "train.csv"
test_path = dataset_asset / "test.csv"
if not all(path.exists() for path in (categories_path, train_path, test_path)):
    raise FileNotFoundError("banking77/ must contain categories.json, train.csv, and test.csv.")

label_names = json.loads(categories_path.read_text(encoding="utf-8"))
label2id = {name: index for index, name in enumerate(label_names)}
id2label = {index: name for name, index in label2id.items()}

def load_csv_split(path):
    frame = pd.read_csv(path)
    if not {"text", "category"}.issubset(frame.columns):
        raise ValueError(f"{path.name} must contain text and category columns.")
    frame = frame[["text", "category"]].copy()
    frame["label"] = frame["category"].map(label2id)
    if frame["label"].isna().any():
        raise ValueError(f"Unknown category in {path.name}.")
    frame["label"] = frame["label"].astype(int)
    return datasets.Dataset.from_pandas(frame[["text", "label"]], preserve_index=False).cast_column(
        "label", datasets.ClassLabel(names=label_names))

raw = DatasetDict({"train": load_csv_split(train_path), "test": load_csv_split(test_path)})
model_source = str(model_asset)
text_col = "text"
label_col = "label"
num_labels = len(label_names)
if num_labels != 77:
    raise ValueError(f"Expected 77 intents, found {num_labels}.")

print(raw)
print("Schema:", raw["train"].features)
print("Official train/test:", len(raw["train"]), len(raw["test"]))
print("Classes:", num_labels)
for row in raw["train"].select(range(3)):
    print(repr(row[text_col]), "->", row[label_col], id2label[row[label_col]])
print(pd.DataFrame({"class_id": range(num_labels), "intent": label_names}).to_string(index=False))

indices = np.arange(len(raw["train"]))
train_idx, val_idx = train_test_split(
    indices, test_size=CFG.validation_size, random_state=CFG.seed,
    stratify=np.asarray(raw["train"][label_col]))
splits = DatasetDict({"train": raw["train"].select(train_idx),
                      "validation": raw["train"].select(val_idx),
                      "test": raw["test"]})
print({k: len(v) for k,v in splits.items()})''')

md("""## 6. Compact exploratory analysis

Word counts provide a readable first approximation of length. The tokenizer-length quantiles below directly validate the token limit.""")
code(r'''train_texts = splits["train"][text_col]
word_lengths = np.array([len(x.split()) for x in train_texts])
class_counts = pd.Series(splits["train"][label_col]).value_counts().sort_index()
eda = {"shortest_words": int(word_lengths.min()), "longest_words": int(word_lengths.max()),
       "mean_words": float(word_lengths.mean()), "median_words": float(np.median(word_lengths))}
print(pd.Series(eda))
print("Smallest classes:\n", pd.DataFrame({"intent": label_names, "count": class_counts.values}).nsmallest(10, "count"))
print("Largest classes:\n", pd.DataFrame({"intent": label_names, "count": class_counts.values}).nlargest(10, "count"))
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(word_lengths, bins=30); axes[0].set(xlabel="Words", ylabel="Queries", title="Query lengths")
axes[1].bar(np.arange(num_labels), class_counts.values); axes[1].set(xlabel="Class ID", ylabel="Training queries", title="Class frequency")
plt.tight_layout(); plt.show()''')

md("""## 7–8. Tokenization and DataLoaders

Tokenization maps text to integer token IDs. The attention mask is 1 for real tokens and 0 for padding, preventing padding from influencing contextual representations. Dynamic padding pads each batch only to its longest example; truncation at 128 tokens is checked against observed lengths rather than assumed.""")
code(r'''tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=CFG.offline)
sample_token_lengths = np.array([len(tokenizer(t, truncation=False)["input_ids"]) for t in raw["train"][text_col]])
print("Token length quantiles:", dict(zip([.5,.9,.95,.99,1.0], np.quantile(sample_token_lengths,[.5,.9,.95,.99,1.0]))))
print(f"Fraction over max_length={CFG.max_length}: {(sample_token_lengths > CFG.max_length).mean():.4%}")

def tokenize_batch(batch):
    return tokenizer(batch[text_col], truncation=True, max_length=CFG.max_length)

tokenized = splits.map(tokenize_batch, batched=True, desc="Tokenizing")
if label_col != "labels": tokenized = tokenized.rename_column(label_col, "labels")
keep = ["input_ids", "attention_mask", "labels"]
tokenized.set_format(type="torch", columns=keep)
collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
generator = torch.Generator().manual_seed(CFG.seed)
loader_args = dict(batch_size=CFG.batch_size, collate_fn=collator,
                   num_workers=CFG.num_workers, pin_memory=device.type=="cuda")
train_loader = DataLoader(tokenized["train"], shuffle=True, generator=generator, **loader_args)
val_loader = DataLoader(tokenized["validation"], shuffle=False, **loader_args)
test_loader = DataLoader(tokenized["test"], shuffle=False, **loader_args)''')

md("""## 9–11. Model, optimization, training, and validation

DistilBERT is a compressed pretrained Transformer encoder. Its pretrained weights encode general language patterns; fine-tuning updates them for banking intents. A randomly initialized classification head maps the contextual representation to 77 logits (unnormalized class scores).

Cross-entropy penalizes low probability on the correct class. Backpropagation computes how each trainable parameter contributed to that loss. AdamW adapts updates per parameter while decoupling weight decay; the scheduler warms up and then gradually lowers the learning rate to stabilize fine-tuning.""")
code(r'''model = AutoModelForSequenceClassification.from_pretrained(
    model_source, num_labels=num_labels, id2label=id2label, label2id=label2id,
    local_files_only=CFG.offline, ignore_mismatched_sizes=True).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total_params:,} total; {trainable_params:,} trainable")

optimizer = AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
total_steps = len(train_loader) * CFG.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(total_steps * CFG.warmup_ratio), total_steps)

def move_batch(batch): return {k:v.to(device, non_blocking=True) for k,v in batch.items()}

def train_one_epoch(model, loader):
    model.train(); loss_sum = 0.0
    bar = tqdm(loader, desc="Training", leave=False)
    for batch in bar:
        batch = move_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        optimizer.step(); scheduler.step()
        loss_sum += loss.item() * batch["labels"].size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return loss_sum / len(loader.dataset)

def evaluate(model, loader):
    model.eval(); loss_sum = 0.0; y_true=[]; y_pred=[]
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = move_batch(batch); outputs = model(**batch)
            loss_sum += outputs.loss.item() * batch["labels"].size(0)
            y_true.extend(batch["labels"].cpu().numpy())
            y_pred.extend(outputs.logits.argmax(1).cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = precision_recall_fscore_support(y_true, y_pred, labels=range(num_labels),
                                               average="macro", zero_division=0)[2]
    return loss_sum/len(loader.dataset), accuracy, macro_f1

history=[]; best_f1=-np.inf; best_epoch=None
for epoch in range(1, CFG.epochs+1):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_accuracy, val_macro_f1 = evaluate(model, val_loader)
    history.append({"epoch":epoch, "train_loss":train_loss, "validation_loss":val_loss,
                    "validation_accuracy":val_accuracy, "validation_macro_f1":val_macro_f1})
    print(f"Epoch {epoch}/{CFG.epochs} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
          f"Val Accuracy {val_accuracy:.4f} | Val Macro F1 {val_macro_f1:.4f}")
    if val_macro_f1 > best_f1:
        best_f1, best_epoch = val_macro_f1, epoch
        model.save_pretrained(BEST_DIR); tokenizer.save_pretrained(BEST_DIR)
        (BEST_DIR/"experiment_metadata.json").write_text(json.dumps({"best_epoch":best_epoch,
            "best_validation_macro_f1":best_f1, "config":asdict(CFG)}, indent=2))

history_df = pd.DataFrame(history)
history_df.to_csv(OUTPUT_DIR/"training_history.csv", index=False)
print(f"Best checkpoint: epoch {best_epoch}, validation macro F1={best_f1:.4f}")''')

md("""## 12. Training history""")
code(r'''fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history_df.epoch, history_df.train_loss, marker="o", label="train")
axes[0].plot(history_df.epoch, history_df.validation_loss, marker="o", label="validation")
axes[0].set(title="Loss", xlabel="Epoch"); axes[0].legend()
axes[1].plot(history_df.epoch, history_df.validation_accuracy, marker="o")
axes[1].set(title="Validation accuracy", xlabel="Epoch")
axes[2].plot(history_df.epoch, history_df.validation_macro_f1, marker="o")
axes[2].set(title="Validation macro F1", xlabel="Epoch")
plt.tight_layout(); plt.show()''')

md("""## 13–15. Reload the best model, collect raw test predictions, then calculate metrics

The prediction loop only collects labels, logits, probabilities, and confidence. Metrics are calculated afterward. Accuracy measures the fraction correct; macro F1 gives every intent equal weight, so weak performance on smaller or difficult classes is not hidden by common classes.""")
code(r'''del model
if device.type == "cuda": torch.cuda.empty_cache()
model = AutoModelForSequenceClassification.from_pretrained(BEST_DIR, local_files_only=True).to(device)

def get_predictions(model, loader):
    model.eval(); labels=[]; logits=[]
    with torch.no_grad():
        for batch in tqdm(loader, desc="Final test prediction"):
            batch = move_batch(batch)
            labels.append(batch["labels"].cpu().numpy())
            logits.append(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.cpu().numpy())
    y_true=np.concatenate(labels); all_logits=np.concatenate(logits)
    shifted=all_logits-all_logits.max(axis=1, keepdims=True)
    probabilities=np.exp(shifted)/np.exp(shifted).sum(axis=1, keepdims=True)
    y_pred=probabilities.argmax(axis=1); confidence=probabilities.max(axis=1)
    return y_true, y_pred, all_logits, probabilities, confidence

y_true, y_pred, test_logits, test_probabilities, confidence = get_predictions(model, test_loader)
np.save(OUTPUT_DIR/"test_logits.npy", test_logits)
np.save(OUTPUT_DIR/"test_probabilities.npy", test_probabilities)

def calculate_metrics_manual(y_true, y_pred, n_classes):
    rows=[]
    for c in range(n_classes):
        tp=int(np.sum((y_true==c)&(y_pred==c)))
        fp=int(np.sum((y_true!=c)&(y_pred==c)))
        fn=int(np.sum((y_true==c)&(y_pred!=c)))
        support=int(np.sum(y_true==c))
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        rows.append((c,id2label[c],support,tp,fp,fn,precision,recall,f1))
    frame=pd.DataFrame(rows, columns=["class_id","intent_name","support","true_positive",
        "false_positive","false_negative","precision","recall","f1"])
    accuracy=float(np.sum(y_true==y_pred)/len(y_true))
    summary={"accuracy":accuracy, "macro_precision":frame.precision.mean(),
             "macro_recall":frame.recall.mean(), "macro_f1":frame.f1.mean(),
             "weighted_f1":float(np.average(frame.f1, weights=frame.support))}
    return summary, frame

metrics, per_class = calculate_metrics_manual(y_true, y_pred, num_labels)
sk_acc=accuracy_score(y_true,y_pred)
sk_macro=precision_recall_fscore_support(y_true,y_pred,labels=range(num_labels),average="macro",zero_division=0)
sk_weighted_f1=precision_recall_fscore_support(y_true,y_pred,labels=range(num_labels),average="weighted",zero_division=0)[2]
checks={"accuracy":sk_acc,"macro_precision":sk_macro[0],"macro_recall":sk_macro[1],
        "macro_f1":sk_macro[2],"weighted_f1":sk_weighted_f1}
print("Manual metrics:", metrics)
print("Sklearn metrics:", checks)
print("Absolute differences:", {k:abs(metrics[k]-checks[k]) for k in metrics})
assert all(np.isclose(metrics[k], checks[k]) for k in metrics)
per_class.to_csv(OUTPUT_DIR/"per_class_metrics.csv",index=False)
print("\nTop 10 intents:\n",per_class.nlargest(10,"f1")[["intent_name","precision","recall","f1","support"]].to_string(index=False))
print("\nBottom 10 intents:\n",per_class.nsmallest(10,"f1")[["intent_name","precision","recall","f1","support"]].to_string(index=False))''')

md("""## 16–18. Confusion, error, and confidence analysis

The full 77×77 matrix is computed without using it for model selection. Tables expose the dominant directional errors; any semantic explanation must be treated as a hypothesis after reading the actual examples.""")
code(r'''confusion=np.zeros((num_labels,num_labels),dtype=int)
for true,pred in zip(y_true,y_pred): confusion[true,pred]+=1
np.save(OUTPUT_DIR/"confusion_matrix.npy",confusion)
pair_rows=[]
for t in range(num_labels):
    for p in range(num_labels):
        if t!=p and confusion[t,p]>0: pair_rows.append((id2label[t],id2label[p],int(confusion[t,p])))
confusion_pairs=pd.DataFrame(pair_rows,columns=["true_intent","predicted_intent","error_count"]).sort_values("error_count",ascending=False)
print(confusion_pairs.head(15).to_string(index=False))

texts=np.asarray(splits["test"][text_col],dtype=object)
true_class_probability=test_probabilities[np.arange(len(y_true)),y_true]
predictions=pd.DataFrame({"text":texts,"true_label_id":y_true,
    "true_intent":[id2label[i] for i in y_true],"predicted_label_id":y_pred,
    "predicted_intent":[id2label[i] for i in y_pred],"confidence":confidence,
    "true_class_probability":true_class_probability,"correct":y_true==y_pred})
predictions.to_csv(OUTPUT_DIR/"test_predictions.csv",index=False)
errors=predictions.loc[~predictions.correct].copy()
errors.to_csv(OUTPUT_DIR/"misclassified_examples.csv",index=False)
print("\n20 highest-confidence errors:\n",errors.nlargest(20,"confidence").to_string(index=False))
print("\n20 lowest-confidence errors:\n",errors.nsmallest(20,"confidence").to_string(index=False))

correct_mask=(y_true==y_pred)
avg_correct=float(confidence[correct_mask].mean()) if correct_mask.any() else float("nan")
avg_incorrect=float(confidence[~correct_mask].mean()) if (~correct_mask).any() else float("nan")
print("Average confidence correct/incorrect:",avg_correct,avg_incorrect)
for threshold in [0.9,0.8,0.7]:
    selected=confidence>=threshold
    print(f">={threshold}: coverage={selected.mean():.3f}, accuracy={correct_mask[selected].mean() if selected.any() else float('nan'):.3f}, n={selected.sum()}")
plt.hist(confidence[correct_mask],bins=20,alpha=.6,label="correct")
plt.hist(confidence[~correct_mask],bins=20,alpha=.6,label="incorrect")
plt.xlabel("Maximum softmax probability");plt.ylabel("Predictions");plt.title("Confidence distributions");plt.legend();plt.show()''')

md("""## 19. Inference function and live examples

Softmax converts logits into probabilities summing to one. These scores are useful for ranking classes but are not guaranteed to be calibrated probabilities.""")
code(r'''def predict_intent(text, model=model, tokenizer=tokenizer, device=device):
    encoded=tokenizer(text,return_tensors="pt",truncation=True,max_length=CFG.max_length)
    encoded={k:v.to(device) for k,v in encoded.items()}
    model.eval()
    with torch.no_grad(): probs=torch.softmax(model(**encoded).logits,dim=-1)[0]
    values,ids=torch.topk(probs,k=3)
    top3=[{"intent":id2label[int(i)],"probability":float(v)} for v,i in zip(values.cpu(),ids.cpu())]
    return {"predicted_intent":top3[0]["intent"],"confidence":top3[0]["probability"],"top_3":top3}

custom_queries=["I withdrew money but got charged extra","My card payment was declined",
"Where is my replacement card?","I do not recognize this cash withdrawal",
"Why is my bank transfer still pending?","Can I change my PIN?",
"The exchange rate on my card purchase looks wrong","I want to close my account",
"My cash deposit is missing","Can I use my card abroad?"]
for query in custom_queries: print("\n",query,"\n",json.dumps(predict_intent(query),indent=2))''')

md("""## 20. Optional TF–IDF baseline

This uses the same train split and untouched test set. It provides evidence for whether contextual Transformer fine-tuning improves over a strong sparse-text baseline.""")
code(r'''comparison_rows=[{"model":"Fine-tuned DistilBERT",**metrics}]
if CFG.run_baseline:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True)
    x_train=vectorizer.fit_transform(splits["train"][text_col]); x_test=vectorizer.transform(splits["test"][text_col])
    baseline=LogisticRegression(max_iter=1000,solver="lbfgs",random_state=CFG.seed)
    baseline.fit(x_train,splits["train"][label_col]); baseline_pred=baseline.predict(x_test)
    base_metrics,_=calculate_metrics_manual(y_true,baseline_pred,num_labels)
    comparison_rows.append({"model":"TF-IDF + Logistic Regression",**base_metrics})
comparison_df=pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUTPUT_DIR/"model_comparison.csv",index=False)
display(comparison_df)''')

md("""## 21. Findings (generated only from measured results)""")
code(r'''mutual=confusion+confusion.T; np.fill_diagonal(mutual,0)
i,j=np.unravel_index(np.argmax(np.triu(mutual,1)),mutual.shape)
hardest=per_class.nsmallest(5,"f1").intent_name.tolist()
print("## Findings")
print(f"Measured: DistilBERT test accuracy was {metrics['accuracy']:.4f} and macro F1 was {metrics['macro_f1']:.4f}.")
print(f"Measured: The five lowest-F1 intents were {hardest}.")
print(f"Measured: {id2label[i]} and {id2label[j]} had {mutual[i,j]} mutual confusions.")
print(f"Measured: mean confidence was {avg_correct:.4f} for correct and {avg_incorrect:.4f} for incorrect predictions.")
if CFG.run_baseline:
    delta=metrics["macro_f1"]-base_metrics["macro_f1"]
    print(f"Measured: DistilBERT minus baseline macro-F1 was {delta:+.4f}.")
print("Interpretation (hypothesis): inspect the saved error rows for recurring vocabulary, ambiguous wording, and neighboring intent definitions before attributing a cause.")

summary={"model":CFG.model_name,"dataset":"Banking77","classes":num_labels,"epochs":CFG.epochs,
 "batch_size":CFG.batch_size,"learning_rate":CFG.learning_rate,"best_epoch":best_epoch,
 "best_validation_macro_f1":best_f1,"test":metrics,"most_confused_pair":[id2label[i],id2label[j]],
 "mutual_confusions":int(mutual[i,j]),"test_errors":int((~correct_mask).sum()),
 "avg_confidence_correct":avg_correct,"avg_confidence_incorrect":avg_incorrect}
(OUTPUT_DIR/"experiment_summary.json").write_text(json.dumps(summary,indent=2))
print("\nFINAL EXPERIMENT SUMMARY\n",json.dumps(summary,indent=2))
print("\nSaved artifacts:",[p.name for p in OUTPUT_DIR.iterdir()])''')

nb={"cells":cells,"metadata":{"accelerator":"GPU","kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
Path("IntentScope_Kaggle.ipynb").write_text(json.dumps(nb,indent=1),encoding="utf-8")
