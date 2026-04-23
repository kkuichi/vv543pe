#Google Colab

#!pip install evaluate
#!pip install transformers

from transformers import set_seed
import numpy as np
import random
import pandas as pd
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import TFBertModel, TFBertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import torch
from datasets import Dataset


SEED = 42
set_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

#----------------------DATA LOAD-------------------
train_df_path = "yourpath/train_new.csv"
val_df_path = "yourpath/val_new.csv"
test_df_path = "yourpath/test_new.csv"

train = pd.read_csv(train_df_path)
val = pd.read_csv(val_df_path)
test = pd.read_csv(test_df_path)



#--------------------MODEL LOAD-------------------
model_path = "google/bert_uncased_L-12_H-768_A-12"

#pripojenie Google Drive
from google.colab import drive
drive.mount('/content/drive')

#tokenizér a model pre klasifikáciu (2 triedy)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=2)



#-------------------bert fine-tune:----------
#zmrazenie všetkých parametrov základného modelu
for name, param in model.base_model.named_parameters():
    param.requires_grad = False

#odomknutie posledných dvoch vrstiev enkódera + pooler
for name, param in model.base_model.named_parameters():
    if "pooler" in name or "encoder.layer.11" in name or "encoder.layer.10" in name:
        param.requires_grad = True


#------------------DATA PREP------------------
def prepare_dataset(df):
    texts = df["textNdesc"].tolist()
    labels = df["label"].tolist()
    #tokenizácia (max 512 tokenov)
    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    encodings["label"] = labels
    return Dataset.from_dict(encodings)

train_dataset = prepare_dataset(train)
val_dataset = prepare_dataset(val)
test_dataset = prepare_dataset(test)

#add padding(ml:512) -> trainer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#-----------------------metrics---------------------
accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(evaluation):
    logits, labels = evaluation
    #logits -> pravdep
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(logits, axis=1)

    #probs[:,1] -> idx, start 0: 0=class0, 1=class1...
    auc = auc_score.compute(prediction_scores=probs[:, 1], references=labels)['roc_auc']
    acc = accuracy.compute(predictions=preds, references=labels)['accuracy']

    return {"Accuracy": acc, "AUC": auc}


training_args = TrainingArguments(
    output_dir = "./bert_classifier",
    learning_rate = 2e-5,
    lr_scheduler_type = "linear",
    warmup_ratio = 0.1,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps = 2,
    #L2 -overfitting
    weight_decay = 0.01,
    num_train_epochs = 5,

    logging_strategy = "epoch",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    #small dataset ~36:64 -> eval_AUC
    metric_for_best_model = "eval_AUC",
    greater_is_better = True,
    seed = SEED,
    data_seed = SEED,
    report_to = "none"
)


trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
)

trainer.train()

test_metrics = trainer.evaluate(test_dataset)
print("test metrics:/n", test_metrics)

#----------------MODEL SAVE---------------
best_model_path = "yourpath"
trainer.save_model(best_model_path)
tokenizer.save_pretrained(best_model_path)