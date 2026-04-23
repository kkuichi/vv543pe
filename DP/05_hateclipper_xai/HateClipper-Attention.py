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
from tqdm import tqdm
import os
import types
from sklearn.metrics import auc

device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = "yourpath"
os.makedirs(save_dir, exist_ok=True)

#------------------DATA LOAD------------------------------
img_folder = Path("yourpath")
train_df_path = "yourpath"
val_df_path = "yourpath"
test_df_path = "yourpath"

train_df = pd.read_csv(train_df_path)
val_df = pd.read_csv(val_df_path)
test_df = pd.read_csv(test_df_path)

#--------------------CLASS: DATASET----------------------
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
        self.save_hyperparameters()
        
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
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        #------------pretrained clip-----------------------
        self.clip = CLIPModel.from_pretrained(clip_pretrained_model, attn_implementation="eager") #!!!!!!!!!!!

        self.image_encoder = copy.deepcopy(self.clip.vision_model) #1111
        self.text_encoder = copy.deepcopy(self.clip.text_model) #11111

        image_map_layers = [nn.Linear(self.image_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=args.drop_probs[0])]
        text_map_layers = [nn.Linear(self.text_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=args.drop_probs[0])]
        for _ in range(1, self.num_mapping_layers):
            image_map_layers.extend([nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=args.drop_probs[0])])
            text_map_layers.extend([nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=args.drop_probs[0])])

        self.image_map = nn.Sequential(*image_map_layers)
        self.text_map = nn.Sequential(*text_map_layers)

        #-----------------fusion strategy-------------------------
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

        if args.freeze_image_encoder:
            for _, p in self.image_encoder.named_parameters():
                p.requires_grad_(False)
        if args.freeze_text_encoder:
            for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

    def forward(self, batch):
        image_features = self.image_encoder(batch['pixel_values']).pooler_output
        image_features = self.image_map(image_features)
        text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)
        image_features = F.normalize(image_features, p=2, dim=1) # [batch_size, d]
        text_features = F.normalize(text_features, p=2, dim=1) # [batch_size, d]

        if self.fusion in ['align', 'align_shuffle']:
            features = torch.mul(image_features, text_features)  # [batch_size, d]
        elif self.fusion.startswith('cross'):
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1))  # [batch_size, d, d]
            features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion}")

        features = self.pre_output(features)
        logits = self.output(features)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        return preds, logits


    def common_step(self, batch, batch_idx, calling_function='val'):
        output = {}
        pixel_values = batch['pixel_values']
        if len(pixel_values.shape) == 3:
            pixel_values = pixel_values.unsqueeze(0)
        image_features = self.image_encoder(pixel_values=pixel_values).pooler_output
        text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)

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

    #------CHANGED-----------------------
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

#--------------------------------------------------------------
torch.serialization.add_safe_globals([types.SimpleNamespace])
model = CLIPClassifier.load_from_checkpoint("yourpath")
model.eval()
print("Model loaded successfully")


save_dir = "/yourpath"
os.makedirs(save_dir, exist_ok=True)
sample_csv_path = "yourpath/test_sample_350.csv"
test_sample_df = pd.read_csv(sample_csv_path)
# ------------------- Dataset + Collator -------------------
dataset = MemeDataset(test_sample_df, img_folder, image_size=224)
collator = CustomCollator(clip_model="openai/clip-vit-base-patch32")
# ------------------- DataLoader -------------------
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)


from transformers import CLIPProcessor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print(processor.image_processor.image_mean)  
print(processor.image_processor.image_std)    

mean = torch.tensor(processor.image_processor.image_mean).view(1, -1, 1, 1).to(device)
std = torch.tensor(processor.image_processor.image_std).view(1, -1, 1, 1).to(device)

# ----------------SETTINGS----------------
N_PERTURB = 100
SIGMA = 0.1

# ----------------ATTENTION ROLLOUT----------------
def attention_rollout(model, x):

    with torch.no_grad():
        outputs = model.image_encoder(
            pixel_values=x,
            output_attentions=True
        )

    attentions = outputs.attentions
    num_tokens = attentions[0].shape[-1]
    num_patches = num_tokens - 1 

    side = int(np.sqrt(num_patches))
    assert side * side == num_patches, f"num patchecs {num_patches} err"
    rollout = torch.eye(num_tokens).to(x.device)

    for attention in attentions:
        attention = attention.mean(dim=1)
        attention = attention + torch.eye(num_tokens).to(x.device)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(attention[0], rollout)
    cls_attention = rollout[0, 1:]
    attn_map = cls_attention.cpu().numpy().reshape(side, side)
    attn_map = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)

    Phi = torch.tensor(attn_map, dtype=torch.float32, device=x.device)

    assert Phi.shape == (224, 224)
    print(f"Min: {Phi.min().item():.4f}, Max: {Phi.max().item():.4f}, "
      f"Mean: {Phi.mean().item():.4f}, Std: {Phi.std().item():.4f}, "
      f"L2 norm: {Phi.norm().item():.4f}")

    return Phi



#-----------------CONFIDENCE------------------

N = 0
indicator_sum = 0

for batch in tqdm(loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    x = batch["pixel_values"]
    with torch.no_grad():
        _, logits = model(batch)
    probs = torch.softmax(logits, dim=1)
    c = torch.argmax(probs, dim=1)
    # y_i^c
    y_i_c = probs[0, c].item()
    # explanation Φ
    Phi = attention_rollout(model, x)
    Phi = Phi.unsqueeze(0).unsqueeze(0)   # (1,1,224,224)
    # masked image
    x_expl = x * Phi
    with torch.no_grad():
        _, logits_expl = model({**batch, "pixel_values": x_expl})
    probs_expl = torch.softmax(logits_expl, dim=1)
    # o_i^c
    o_i_c = probs_expl[0, c].item()
    # indicator
    if y_i_c < o_i_c:
        indicator_sum += 1
    N += 1
CI = (indicator_sum / N) * 100
print("CI mean: ",CI)



#-------------ROBUSTNESS-----------------
epsilon = 0.05
K = 10
L_tilde_values = []

for batch in tqdm(loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    x_i = batch["pixel_values"]
    # f(x_i)  -> explanation
    Phi_i = attention_rollout(model, x_i)
    max_ratio = 0.0
    for _ in range(K):
        #generate x_j ∈ Nε(x_i)
        noise = torch.randn_like(x_i) * epsilon
        x_j = torch.clamp(x_i + noise, 0, 1)
        # ||x_i - x_j||_2
        input_dist = torch.norm((x_i - x_j).view(-1), p=2)
        if input_dist == 0:
            continue
        # f(x_j)
        Phi_j = attention_rollout(model, x_j)
        # ||f(x_i) - f(x_j)||_2
        explanation_dist = torch.norm((Phi_i - Phi_j).view(-1), p=2)
        ratio = explanation_dist / input_dist
        if ratio > max_ratio:
            max_ratio = ratio.item()

    L_tilde_values.append(max_ratio)
L_tilde_values = np.array(L_tilde_values)
print("Robustness: ",L_tilde_values.mean())




#-------------FAITHFULNESS---------------------------
N_STEPS = 50
faithfulness_values = []

for batch in tqdm(loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    x = batch["pixel_values"]
    # f(x)
    with torch.no_grad():
        _, logits = model(batch)
    target_class = torch.argmax(logits, dim=1)
    f_x = logits[0, target_class].item()
    # Φ
    Phi = attention_rollout(model, x)
    Phi_flat = Phi.flatten()
    #deletion seq
    indices = torch.argsort(Phi_flat, descending=True)
    num_pixels = Phi_flat.shape[0]
    step = max(1, num_pixels // N_STEPS)
    mask = torch.ones(num_pixels, device=device)
    f_values = [f_x]
    removed = 0
    for k in range(0, num_pixels, step):
        current_step = min(step, num_pixels - removed)
        if current_step <= 0:
            break

        remove_idx = indices[removed:removed + current_step]
        mask[remove_idx] = 0
        mask_img = mask.view(1, 1, 224, 224)
        x_k = x * mask_img
        with torch.no_grad():
            _, logits_k = model({**batch, "pixel_values": x_k})

        f_xk = logits_k[0, target_class].item()
        f_values.append(f_xk)
        removed += current_step
    # deletion curve
    f_values = np.array(f_values)
    x_axis = np.linspace(0, 1, len(f_values))
    # AUC
    faithfulness = auc(x_axis, f_values)
    faithfulness_values.append(faithfulness)

faithfulness_values = np.array(faithfulness_values)
print("Faithfulness del mean:",faithfulness_values.mean())



# ---------------- INFIDELITY ----------------

def compute_infidelity(Phi, batch, model, target_class):

    x = batch["pixel_values"]
    Phi_flat = Phi.flatten()

    with torch.no_grad():
        _, logits = model(batch)

    f_x = logits[:, target_class]
    baseline_norm = (0.5-mean)/std
    approx_list = []
    diff_list = []

    for _ in range(N_PERTURB):
        epsilon = torch.randn_like(x) * SIGMA
        baseline_noisy = baseline_norm + epsilon 

        I = x - baseline_noisy
        x_pert = baseline_noisy

        perturbed_batch = dict(batch)
        perturbed_batch["pixel_values"] = x_pert

        with torch.no_grad():
            _, logits_pert = model(perturbed_batch)
        f_x_pert = logits_pert[:, target_class]
        diff = f_x - f_x_pert
        I_gray = I.mean(dim=1) 
        I_flat = I_gray.flatten()
        approx = (I_flat * Phi_flat).sum() 
        
        approx_list.append(approx)
        diff_list.append(diff)

    approx_t = torch.tensor(approx_list, device=device)
    diff_t = torch.tensor(diff_list, device=device)

        #infd = (approx - diff) ** 2
        #infd = infd.item()
    num = (approx_t * diff_t).sum()
    den = (approx_t ** 2).sum()
    beta = num / (den + 1e-8)

    #norm  infidelity
    infd_beta = ((beta * approx_t - diff_t) ** 2).mean().item()

    if np.isnan(infd_beta) or np.isinf(infd_beta):
        return np.nan
    return infd_beta #infidelity_sum / valid

#----------------MAIN LOOP----------------
all_scores = []
for batch_idx, batch in enumerate(tqdm(loader)):
    try:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if k != "labels"
        }
        x = batch["pixel_values"]
        with torch.no_grad():
            preds, logits = model(batch)
        target_class = torch.argmax(logits, dim=1).item()
        Phi = attention_rollout(model, x)
        #print("Phi: ", Phi.shape)
        infd = compute_infidelity(
            Phi,
            batch,
            model,
            target_class
        )
        if not np.isnan(infd):
            all_scores.append(infd)
    except Exception as e:
        print(f"Error on sample {batch_idx}: {e}")
        continue


# ---------------- RESULTS ----------------
all_scores = np.array(all_scores)

print("\n============================")
print("FINAL INFIDELITY RESULTS")
print("============================")

print("Images processed:", len(all_scores))
print("Mean Attention:", all_scores.mean())




