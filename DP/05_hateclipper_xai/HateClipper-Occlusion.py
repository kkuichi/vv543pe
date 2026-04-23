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
from scipy.spatial.distance import cdist

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
	#-----------------ADDED NEW API----------------------
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
        #-----CHANGED
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
        #----new-----(for cm)
        output['preds'] = preds
        output['preds_proxy'] = preds_proxy
        output['labels'] = batch['labels']
        #------------(for cm)
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
        #new-----------------(for cm)
        self.test_step_outputs.append({
            'preds': output['preds'].detach().cpu(),
            'labels': batch['labels'].detach().cpu(),
            'probs': output['preds_proxy'].detach().cpu()
          })
        #---------------------(for cm)
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
           #new--------for cm
            preds = torch.cat([x['preds'] for x in self.test_step_outputs])
            labels = torch.cat([x['labels'] for x in self.test_step_outputs])
            cm = torchmetrics.functional.confusion_matrix(preds, labels, task='binary')
            self.print("Confusion matrix:\n", cm)
           #new---------for cm
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


# ---------------- SETTINGS ----------------

N_PERTURB = 100
PATCH_SIZE = 32
STRIDE = 16
SIGMA = 0.1

# ----------------OCCLUSION----------------
def occlusion_phi(model, batch, patch_size=32):
    model.eval()
    device = batch["pixel_values"].device
    x = batch["pixel_values"]
    B, C, H, W = x.shape
    assert H == 224 and W == 224

    #confidence target class
    with torch.no_grad():
        _, base_logits = model(batch)
        base_score = torch.sigmoid(base_logits)[0]   #probability target class

    #smaller map sizes
    h_steps = (H + patch_size - 1) // patch_size   # ceil(H / patch_size)
    w_steps = (W + patch_size - 1) // patch_size   # ceil(W / patch_size)
    small_map = torch.zeros((h_steps, w_steps), device=device)

    #patches
    for i in range(h_steps):
        for j in range(w_steps):
            y = i * patch_size
            x_pos = j * patch_size
            y_end = min(y + patch_size, H)
            x_end = min(x_pos + patch_size, W)

            occluded = x.clone()
            occluded[:, :, y:y_end, x_pos:x_end] = (0.5 - mean) / std

            new_batch = dict(batch)
            new_batch["pixel_values"] = occluded

            with torch.no_grad():
                _, logits = model(new_batch)
                score = torch.sigmoid(logits)[0]

            #importance = 1 - conf after occlusion
            small_map[i, j] = 1 - score

    #orig map sizes
    small_np = small_map.detach().cpu().numpy()
    Phi = cv2.resize(small_np, (W, H), interpolation=cv2.INTER_CUBIC)
    Phi = torch.tensor(Phi, dtype=torch.float32, device=device)
#    print(f"Min: {Phi.min().item():.4f}, Max: {Phi.max().item():.4f}, "
#      f"Mean: {Phi.mean().item():.4f}, Std: {Phi.std().item():.4f}, "
#      f"L2 norm: {Phi.norm().item():.4f}")
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
all_images = []      #norm tensors from batcc (1,3,224,224)
all_phis = []        #importance maps (224,224)

print("Robustness calculation")
for batch in tqdm(loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    x = batch["pixel_values"]               # (1,3,224,224)
    all_images.append(x.squeeze(0).cpu().numpy())   # (3,224,224), numpy
    with torch.no_grad():
        Phi = occlusion_phi(model, batch)           # (224,224) tensor
    all_phis.append(Phi.cpu().numpy())              #save as numpy

all_images = np.array(all_images)        # (N, 3, 224, 224)
all_phis = np.array(all_phis)            # (N, 224, 224)

N = len(all_images)
print(f"Collected {N} images and explanations.")

# ------------------explanation norm(L2) ------------------
print("Normalizing explanations...")
phi_norms = np.linalg.norm(all_phis.reshape(N, -1), axis=1, keepdims=True)
phi_norms[phi_norms == 0] = 1.0  #protect: /0
all_phis_norm = all_phis / phi_norms.reshape(N, 1, 1)

# ------------------choose ε ------------------
#img to vec
img_vectors = all_images.reshape(N, -1)
dist_img = cdist(img_vectors, img_vectors, metric='euclidean')   # (N,N)

#choose % not null
non_zero_dist = dist_img[dist_img > 0]
epsilon = np.percentile(non_zero_dist, 5)
print(f"Chosen epsilon = {epsilon:.4f}")

# ------------------calculate L for every x_i ------------------
L_tilde = []
print("Computing robustness...")
for i in tqdm(range(N)):
    #neighboors in radius ε (include x_i)
    neighbors = np.where(dist_img[i] <= epsilon)[0]
    neighbors = neighbors[neighbors != i]   #exclude x_i
    if len(neighbors) == 0:
        continue
    max_ratio = 0.0
    for j in neighbors:
        dist = dist_img[i, j]
        if dist == 0:
            continue
        phi_diff = np.linalg.norm(all_phis[i] - all_phis[j])
        ratio = phi_diff / dist
        if ratio > max_ratio:
            max_ratio = ratio
    L_tilde.append(max_ratio)

robustness = np.mean(L_tilde)
print(f"\nRobustness (ε={epsilon:.4f}): {robustness:.6f}")






#-------------FAITHFULNESS---------------------------
def deletion_curve_values(model, batch, Phi, step_ratio=0.1):

   model.eval()
   x = batch["pixel_values"]
   #after batch load
   print("Pixel values min:", batch["pixel_values"].min().item())
   print("Pixel values max:", batch["pixel_values"].max().item())

  # print("\n--- NEW SAMPLE ---")
  # print("Phi shape:", Phi.shape)
  # print("x shape:", x.shape)

   with torch.no_grad():
       _, logits = model(batch)

    # sigmoid for binar
   print("Logits shape:", logits.shape)
   score_original = torch.sigmoid(logits[0, 0]).item()
   print("Original score:", score_original)

    #flatten
   Phi_flat = Phi.flatten()
   indices = torch.argsort(Phi_flat, descending=True)
   num_pixels = Phi_flat.shape[0]
   step = max(1, int(step_ratio * num_pixels))
   mask = torch.ones(num_pixels, device=x.device)

   baseline = (0.5 - mean) / std

   f_values = [score_original]
   removed = 0

   while removed < num_pixels:

       current_step = min(step, num_pixels - removed)
       remove_idx = indices[removed:removed + current_step]
       mask[remove_idx] = 0
       mask_img = mask.view(1, 1, *Phi.shape)
       x_k = x * mask_img + (1 - mask_img) * baseline
       x_k = torch.clamp(x_k, 0, 1)
       with torch.no_grad():
           _, logits_k = model({**batch, "pixel_values": x_k})
       score_k = torch.sigmoid(logits_k[0, 0]).item()
       f_values.append(score_k)
       removed += current_step

   print("Logits:", logits[0])
   print("ORIG sum:", x.sum().item())
   print("MASKED sum:", x_k.sum().item())
   print("Original prob:", f_values[0])
   print("Final prob:", f_values[-1])

   return np.array(f_values)


#--------------MAIN CALCULATION---------------
def compute_deletion_statistics(model, loader, occlusion_phi_function, device, save_path,):

   os.makedirs(save_path, exist_ok=True)
   csv_path = os.path.join(save_path, "deletion_results_occl_1903.csv")
   all_aucs_raw = []
   all_curves_raw = []
   with open(csv_path, "w", newline="") as f:
       writer = csv.writer(f)
       writer.writerow([
           "sample_id",
           "auc_raw",
           "curve_raw"
       ])

       for i, batch in enumerate(tqdm(loader, desc="Computing deletion")):
           batch = {k: v.to(device) for k, v in batch.items()}
           Phi = occlusion_phi_function(model, batch)
           curve = deletion_curve_values(model, batch, Phi)
           x_axis = np.linspace(0, 1, len(curve))
           # RAW AUC
           auc_raw = auc(x_axis, curve)
           if i < 3:
               print("\nSample", i)
               print("Curve raw:", curve[:10])
               print("AUC raw:", auc_raw)

           all_aucs_raw.append(auc_raw)
           all_curves_raw.append(curve)
           writer.writerow([i, auc_raw]) #, auc_norm])

   all_aucs_raw = np.array(all_aucs_raw)
   print("\n========== RESULTS ==========")
   print("RAW AUC Mean:", all_aucs_raw.mean(), "Std:", all_aucs_raw.std())
   print("=============================\n")


aucs_raw = compute_deletion_statistics(
       model=model,
       loader=loader,
       occlusion_phi_function=occlusion_phi,
       device=device,
       save_path="yourpath"
   )







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
  #     approx = (I_flat * Phi_flat).sum() 
       approx_list.append(approx)
       diff_list.append(diff)
   approx_t = torch.tensor(approx_list, device=device)
   diff_t = torch.tensor(diff_list, device=device)
   num = (approx_t * diff_t).sum()
   den = (approx_t ** 2).sum()
    beta = num / (den + 1e-8)
    infd_beta = ((beta * approx_t - diff_t) ** 2).mean().item()
    if np.isnan(infd_beta) or np.isinf(infd_beta):
       return np.nan
    return infd_beta

#Main Calculation
all_scores = []
for batch_idx, batch in enumerate(tqdm(loader)):
   try:
       batch = {
           k: v.to(device)
           for k, v in batch.items()
           if k != "labels"
       }
       with torch.no_grad():
           preds, logits = model(batch)
       target_class = torch.argmax(logits, dim=1).item()
       Phi = occlusion_phi(model, batch)
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


#---------------- RESULTS ----------------
all_scores = np.array(all_scores)

print("\n============================")
print("FINAL INFIDELITY RESULTS")
print("============================")

print("Images processed:", len(all_scores))
print("Mean Occlusion:", all_scores.mean())


