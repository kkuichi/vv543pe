#PERUN

import copy
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import clip
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random
from transformers import set_seed, CLIPProcessor, CLIPTokenizer, CLIPModel
from sklearn.metrics import f1_score
from types import SimpleNamespace
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from captum.attr import IntegratedGradients
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = "yourpath"
os.makedirs(save_dir, exist_ok=True)

#------------------DATA LOAD------------------------------
img_folder = Path("yourpath")
train_df_path = "yourpath/train_new.csv"
val_df_path = "yourpath/val_new.csv"
test_df_path = "yourpath/test_new.csv"

train_df = pd.read_csv(train_df_path)
val_df = pd.read_csv(val_df_path)
test_df = pd.read_csv(test_df_path)

#--------------------CLASS: DATASET----------------------
"""
    vlastná implementácia datasetu pre multimodálne učenie:
    každý vzor obsahuje: obrázok + text + label
    """

class MemeDataset(Dataset):
    def __init__(self, df, img_folder, image_size=224):
        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        item['image'] = Image.open(self.img_folder / row['img']).convert('RGB').resize((self.image_size, self.image_size))
        item['text'] = row['text']
        item['label'] = int(row['label'])
        return item

#--------------------CUSTOM COLLATOR----------------------
"""
    Zodpovedá za: batchovanie + tokenizáciu textu + konverziu na torch tensory
    CLIP vyžaduje: pixel_values (image tensor) + input_ids + attention_mask (text tensor)
"""

class CustomCollator:
    def __init__(self, clip_model="openai/clip-vit-base-patch32"):
        self.image_processor = CLIPProcessor.from_pretrained(clip_model)
        self.text_processor = CLIPTokenizer.from_pretrained(clip_model)

    def __call__(self, batch):
        pixel_values = self.image_processor(images=[item['image'] for item in batch], return_tensors="pt")['pixel_values']
        text_output = self.text_processor([item['text'] for item in batch], padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        batch_out = {
            'pixel_values': pixel_values,
            'input_ids': text_output['input_ids'],
            'attention_mask': text_output['attention_mask'],
            'labels': labels
        }

        return batch_out

#--------------------CLASSIFIER----------------------
class CLIPClassifier(pl.LightningModule):

    def __init__(self, args, clip_pretrained_model="openai/clip-vit-base-patch32"):
        super().__init__()
        self.save_hyperparameters() #uloženie hyperparametrov do checkpointu (Lightning feature)
        # hyperparametre
        self.num_mapping_layers = args.num_mapping_layers
        self.map_dim = args.map_dim
        self.fusion = args.fusion
        self.num_pre_output_layers = args.num_pre_output_layers
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.weight_image_loss = args.weight_image_loss
        self.weight_text_loss = args.weight_text_loss
        #--------------metrics----------------------------
        self.acc = torchmetrics.Accuracy(task="binary")
        self.auroc = torchmetrics.AUROC(task="binary")
	    #-----------------ADDED NEW API----------------------
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        #------------pretrained clip-----------------------
        #načítanie predtrénovaného CLIP modelu
        self.clip = CLIPModel.from_pretrained(clip_pretrained_model, attn_implementation="eager") #!!!!!!!!!!!

        #separácia obrazovej a textovej časti CLIP modelu
        self.image_encoder = copy.deepcopy(self.clip.vision_model) #1111
        self.text_encoder = copy.deepcopy(self.clip.text_model) #11111

        #--------MAPPING---------------------
        image_map_layers = [nn.Linear(self.image_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=args.drop_probs[0])]
        text_map_layers = [nn.Linear(self.text_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=args.drop_probs[0])]
        for _ in range(1, self.num_mapping_layers):
            image_map_layers.extend([nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=args.drop_probs[0])])
            text_map_layers.extend([nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=args.drop_probs[0])])

        self.image_map = nn.Sequential(*image_map_layers)
        self.text_map = nn.Sequential(*text_map_layers)

        #-----------------fusion strategy-------------------------
        #align = element-wise -> dim = map_dim
        #cross = outer product -> dim = map_dim^2
        if args.fusion in ['align', 'align_shuffle']:
            pre_output_input_dim = self.map_dim
        elif args.fusion.startswith('cross'):
            pre_output_input_dim = self.map_dim**2

        #-----------------pre output-------------------------
        pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
        output_input_dim = pre_output_input_dim
        if self.num_pre_output_layers >= 1: # first pre-output layer
            pre_output_layers.extend([nn.Linear(pre_output_input_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
            output_input_dim = self.map_dim 
        for _ in range(1, self.num_pre_output_layers): # next pre-output layers
            pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.output = nn.Linear(output_input_dim, 1)

        
        if self.weight_image_loss > 0:
            pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
            for _ in range(self.num_pre_output_layers):
                pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
            self.pre_output_image = nn.Sequential(*pre_output_layers)
            self.output_image = nn.Linear(output_input_dim, 1)

        if self.weight_text_loss > 0:
             pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
             for _ in range(self.num_pre_output_layers):
                 pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
             self.pre_output_text = nn.Sequential(*pre_output_layers)
             self.output_text = nn.Linear(output_input_dim, 1)

        #-----------------loss-------------------------
        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        #------------------FREEZING------------
"""váhy sa NEUPRAVUJÚ počas tréningu, 
   tréning prebieha len na nových vrstvách (mapping + fusion)
"""
        if args.freeze_image_encoder:
            for _, p in self.image_encoder.named_parameters():
                p.requires_grad_(False)
        if args.freeze_text_encoder:
            for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

    def forward(self, batch):
        #extrakcia vizuálnych príznakov z CLIP vision encoderu
        image_features = self.image_encoder(batch['pixel_values']).pooler_output
        image_features = self.image_map(image_features)
        
        #extrakcia textových príznakov z CLIP text encoderu
        text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)
        
        #normalizácia vektorov
        image_features = F.normalize(image_features, p=2, dim=1) # [batch_size, d]
        text_features = F.normalize(text_features, p=2, dim=1) # [batch_size, d]

        #--------------------FUSION LOGIC-------------------
        if self.fusion in ['align', 'align_shuffle']:
            features = torch.mul(image_features, text_features)  # [batch_size, d]
        elif self.fusion.startswith('cross'):
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1))  # [batch_size, d, d]
            features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion}")

        features = self.pre_output(features)
        logits = self.output(features)

        #binárna predikcia (threshold 0.5)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        return preds, logits


    def common_step(self, batch, batch_idx, calling_function='val'):
        output = {}

        #----------FIX INPUT SHAPE-----------
        #ak batch má shape [C, H, W] namiesto [B, C, H, W]
        pixel_values = batch['pixel_values']
        if len(pixel_values.shape) == 3:
            pixel_values = pixel_values.unsqueeze(0)
        
        image_features = self.image_encoder(pixel_values=pixel_values).pooler_output
        text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)

        #normalizácia
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)


        if self.weight_image_loss > 0:
            features_pre_output = self.pre_output_image(image_features)
            logits_image = self.output_image(features_pre_output).squeeze(dim=1)
            preds_proxy_image = torch.sigmoid(logits_image)
            preds_image = (preds_proxy_image >= 0.5).long()

        output['image_loss'] = self.cross_entropy_loss(logits_image, batch['labels'].float())
        output['image_accuracy'] = self.acc(preds_image, batch['labels'])
        output['image_auroc'] = self.auroc(preds_proxy_image, batch['labels'])
        output['preds_image'] = preds_image
        output['probs_image'] = preds_proxy_image

        if self.weight_text_loss > 0:
            features_pre_output = self.pre_output_text(text_features)
            logits_text = self.output_text(features_pre_output).squeeze(dim=1)
            preds_proxy_text = torch.sigmoid(logits_text)
            preds_text = (preds_proxy_text >= 0.5).long()

        output['text_loss'] = self.cross_entropy_loss(logits_text, batch['labels'].float())
        output['text_accuracy'] = self.acc(preds_text, batch['labels'])
        output['text_auroc'] = self.auroc(preds_proxy_text, batch['labels'])
        output['preds_text'] = preds_text
        output['probs_text'] = preds_proxy_text


        if self.fusion in ['align', 'align_shuffle']:
            features = torch.mul(image_features, text_features)
        elif self.fusion.startswith('cross'):
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1))  # outer product [batch_size, d, d]
            features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion}")

        features_pre_output = self.pre_output(features)
        logits = self.output(features_pre_output).squeeze(dim=1)  # [batch_size, 1]



        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()

        output['loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
        output['accuracy'] = self.acc(preds, batch['labels'])
        output['auroc'] = self.auroc(preds_proxy, batch['labels'])

        return output

    def training_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx, calling_function='train')
        self.training_step_outputs.append(output)

        image_loss = output['image_loss'] if self.weight_image_loss > 0 else 0.0
        text_loss = output['text_loss'] if self.weight_text_loss > 0 else 0.0
        total_loss = (output['loss']+self.weight_image_loss*image_loss+self.weight_text_loss*text_loss)
        
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss', output['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', output['accuracy'], on_step=False, on_epoch=True)
        self.log('train/auroc', output['auroc'], on_step=False, on_epoch=True)
        
        if self.weight_image_loss > 0:
            self.log('train/image_loss', image_loss)
        if self.weight_text_loss > 0:
            self.log('train/text_loss', text_loss)
        return total_loss 

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx, calling_function='val')
        self.validation_step_outputs.append(output)
        
        image_loss = output['image_loss'] if self.weight_image_loss > 0 else 0
        text_loss = output['text_loss'] if self.weight_text_loss > 0 else 0
        total_loss = (output['loss']+self.weight_image_loss*image_loss+self.weight_text_loss*text_loss)
        
        self.log('val/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/loss', output['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', output['accuracy'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/auroc', output['auroc'], on_step=False, on_epoch=True, prog_bar=True)
        
        if self.weight_image_loss > 0:
            self.log('train/image_loss', image_loss)
        if self.weight_text_loss > 0:
            self.log('train/text_loss', text_loss)
        return total_loss

    def test_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx, calling_function='test')
        self.log('test/accuracy', output['accuracy'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/auroc', output['auroc'], on_step=False, on_epoch=True, prog_bar=True)
        return output

        def on_train_epoch_end(self):
            self.acc.reset()
            self.auroc.reset()
            self.precision_score.reset()
            self.recall.reset()
            self.f1.reset()
            if hasattr(self, 'training_step_outputs'):
                self.training_step_outputs.clear()

        def on_validation_epoch_end(self):
            self.acc.reset()
            self.auroc.reset()
            self.precision_score.reset()
            self.recall.reset()
            self.f1.reset()
            if hasattr(self, 'validation_step_outputs'):
                self.validation_step_outputs.clear()

        def on_test_epoch_end(self):
            self.acc.reset()
            self.auroc.reset()
            self.precision_score.reset()
            self.recall.reset()
            self.f1.reset()
            if hasattr(self, 'test_step_outputs'):
                self.test_step_outputs.clear()

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
            ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

def create_model(args):
    model = CLIPClassifier(args=args)
    return model

#--------------------data prep-------------------------------
train_dataset = MemeDataset(train_df, img_folder, image_size=224)
val_dataset = MemeDataset(val_df, img_folder, image_size=224)
test_dataset = MemeDataset(test_df, img_folder, image_size=224)
collator = CustomCollator(clip_model="openai/clip-vit-base-patch32")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collator)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, collate_fn=collator)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collator)

#--------------------MODEL PARAM-------------------------------
args = SimpleNamespace(
    weight_image_loss=1.0,
    weight_text_loss=1.0,
    num_mapping_layers=1,
    map_dim=768,
    fusion='cross',  #align/cross
    num_pre_output_layers=1,
    lr=1e-4,
    weight_decay=1e-4,
    accelerator="auto",
    devices="auto",
    max_epochs=5,
    max_steps=-1,
    gradient_clip_val=0.1,
    log_every_n_steps=50,
    val_check_interval=1.0,
    strategy="auto",
    limit_train_batches=1.0,
    limit_val_batches=1.0,
    drop_probs=[0.1, 0.2, 0.3],
    freeze_image_encoder=True,
    freeze_text_encoder=True
)

#-----------TRAIN + SAVE --------------

model = create_model(args)
seed_everything(42, workers=True)

checkpoint_callback = ModelCheckpoint(
    dirpath="yourpath",
    filename="HateClipper-CheckPoint-1103",
    monitor="val/auroc",
    mode="max",
    save_top_k=1,
)

trainer = Trainer(
    accelerator=args.accelerator,
    devices=args.devices,
    max_epochs=args.max_epochs,
    max_steps=args.max_steps,
    callbacks=[checkpoint_callback],
    gradient_clip_val=args.gradient_clip_val,
    log_every_n_steps=args.log_every_n_steps,
    val_check_interval=args.val_check_interval,
    strategy=args.strategy,
    limit_train_batches=args.limit_train_batches,
    limit_val_batches=args.limit_val_batches,
    deterministic=True
)

trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
trainer.test(model=model, dataloaders=test_loader, ckpt_path=best_ckpt_path)
#trainer.test(model=model, dataloaders=test_loader)

