from lime.lime_text import LimeTextExplainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertForSequenceClassification, BertTokenizer
import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
from captum.attr import IntegratedGradients
import random

#-------------------------SEED-------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

#--------------------------CONF--------------------------------
save_path = "yourpath/bert_full_250126"
df_path = "yourpath/test_sample_350_stratified.csv"
output_path = "yourpath/M1_naopc_scores.csv"

TOP_K = 20
BATCH_SIZE = 32
IG_STEPS = 50
BUFFER_SIZE = 10 #buffer pred zapisom do csv


#----------------------MODEL--------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(save_path)
tokenizer = BertTokenizer.from_pretrained(save_path)
model.to(device)
#inference režim (dropout off)
model.eval()
print("[!model loaded!]")

#------------------DATA LOAD------------------------------
df = pd.read_csv(df_path)
df = df.dropna(subset=["text"])
#df["text"] = df["text"].astype(str)
#df = df.head(3)
print("[!data loaded!]")

#----------------------PREDICT---------------------------
#rýchla predikcia pre viac textov naraz (LIME, IG)
def predict_proba(texts):
    probs_all = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_all.append(probs)
    return np.vstack(probs_all)

#predikcia pre SHAP
def predict_proba_shap(texts):
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    texts = [str(t) for t in texts]
    inputs = tokenizer(texts, padding=True, truncation=True,return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()


#pravdepodobnosť pre konkrétnu triedu
def get_prob(input_ids, target):
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        return torch.softmax(logits, dim=1)[0, target].item()


#-----------------------EXPLANATIONS-------------------------------
lime_explainer = LimeTextExplainer(class_names=["non-toxic", "toxic"], split_expression=r'\W+', bow=False)
masker = shap.maskers.Text(tokenizer, mask_token=tokenizer.mask_token)
shap_explainer = shap.Explainer(predict_proba_shap, masker, output_names=["non-toxic", "toxic"])

#-----------------------WORD MAP---------------------------
#mapuje slová -> tokeny
def build_word_map(text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]

    words = text.split()
    word_map = []
    idx = 1
    for word in words:
        n = len(tokenizer.tokenize(word))
        word_map.append(list(range(idx, idx + n)))
        idx += n

    if sum(len(x) for x in word_map) != len(token_ids):
        return None, None

    return input_ids, word_map


#agregácia tokenov -> slová
def token_to_word_attr(token_attr, word_map):
    return np.array([
        np.sum([token_attr[i] for i in indices if i < len(token_attr)])
        for indices in word_map
    ])


######### NAOPC #########

#--------------------------CURVES----------------------------
# sleduje pokles pravdepodobnosti pri maskovaní dôležitých slov
def compute_curve(input_ids, word_map, ranking, target):
    masked = input_ids.clone()
    original = get_prob(masked.unsqueeze(0), target)

    diffs = []
    for idx in ranking:
        masked[word_map[idx]] = tokenizer.mask_token_id
        prob = get_prob(masked.unsqueeze(0), target)
        diffs.append(original - prob)

    return np.array(diffs), original


#-----------------------SUFFICIENCY---------------------------
#test: či samotné vybrané slová stačia na predikciu
def compute_sufficiency(input_ids, word_map, ranking, target):
    masked = torch.full_like(input_ids, tokenizer.mask_token_id)
    for idx in ranking:
        masked[word_map[idx]] = input_ids[word_map[idx]]
    original = get_prob(input_ids.unsqueeze(0), target)
    prob = get_prob(masked.unsqueeze(0), target)
    return original - prob


#------------------------LIMITS-------------------------------
#(NAOPC normalization)
def compute_limits(input_ids, word_map, word_attr, target):
    K = min(TOP_K, len(word_attr))
    best = np.argsort(-np.abs(word_attr))[:K]
    worst = np.argsort(np.abs(word_attr))[:K]
    best_curve, original = compute_curve(input_ids, word_map, best, target)
    worst_curve, _ = compute_curve(input_ids, word_map, worst, target)
    # FULL MASK
    masked_all = input_ids.clone()
    for indices in word_map:
        masked_all[indices] = tokenizer.mask_token_id
    all_mask_prob = get_prob(masked_all.unsqueeze(0), target)
    all_mask_diff = original - all_mask_prob
    max_value = np.sum(best_curve)
    min_value = np.sum(worst_curve)
    upper = (max_value + all_mask_diff) / K
    lower = (min_value + all_mask_diff) / K
    return upper, lower


#------------------------ATTRIBUTIONS--------------------------
#LIME vysvetlenie
def get_lime_attr(text, target, input_ids):
    print(f"Calling get_lime_attr")
    try:
        exp = lime_explainer.explain_instance(text, predict_proba, num_features=len(text.split()), labels=[target])
        print(f"[LIME] explain_instance finished")
        weights = dict(exp.as_list(label=target))
        print(f"[LIME] got weights for {len(weights)} words")
    except Exception as e:
        print(f"LIME error: {e}")
        return None
    token_attr = np.zeros(len(input_ids))
    words = text.split()
    idx = 1
    for word in words:
        n = len(tokenizer.tokenize(word))
        token_attr[idx:idx+n] = weights.get(word, 0.0)
        idx += n
    print(f"LIME succeeded")
    return token_attr


#SHAP vysvetlenie
def get_shap_attr(text, target, input_ids):
    print(f"[SHAP] explain_instance finished")
    try:
        if not isinstance(text, str):
            text = str(text)
        shap_vals = shap_explainer([text])
        print(f"    [SHAP] shap vals")
        # shap_vals.values: (1, num_tokens, num_classes)
        token_attr = shap_vals.values[0, :, target]
        print(f"[SHAP] got token_attr for {len(token_attr)} words")
        # align 
        if len(token_attr) != len(input_ids):
            min_len = min(len(token_attr), len(input_ids))
            token_attr = token_attr[:min_len]
        if np.allclose(token_attr, 0):
            return None
        return token_attr
    except Exception as e:
        print(f"SHAP error: {e}")
        return None



#IG vysvetlenie
def get_ig_attr(input_ids, target):
    print(f"    Calling get_ig_attr")
    try:
        input_ids_t = torch.tensor([input_ids], device=device)
        attn = torch.ones_like(input_ids_t)
        embed = model.get_input_embeddings()
        emb = embed(input_ids_t)
        base = embed(torch.full_like(input_ids_t, tokenizer.mask_token_id))
        def forward(e):
            return model(inputs_embeds=e, attention_mask=attn).logits
        ig = IntegratedGradients(forward)
        print(f"[IG] forward finished")
        attr, _ = ig.attribute(emb, baselines=base, target=target, n_steps=IG_STEPS, return_convergence_delta=True)
        print(f"IG succeeded")
        return attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    except Exception as e:
        print(f"IG error: {e}")
        return None

#-----------------------MAIN--------------------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)

if not os.path.exists(output_path):
    pd.DataFrame(columns=["id", "method", "comp", "suff", "naopc"]).to_csv(output_path, index=False)

methods = ["lime", "shap", "ig"]

results_buffer = []
valid_counts = {m: 0 for m in methods}
invalid_counts = {m: 0 for m in methods}

pbar = tqdm(total=len(df) * len(methods), desc="Progress")

for idx in df.index:
    print(f"\nProcessing idx {idx}")
    text = df.loc[idx, "textNdesc"]
    label = int(df.loc[idx, "label"])
    input_ids, word_map = build_word_map(text)
    if input_ids is None:
        print(f"  -> build_word_map failed")
        for method in methods:
            invalid_counts[method] += 1
            pbar.update(1)
        continue
    else:
        print(f"  -> build_word_map success, input_ids length {len(input_ids)}")
    input_ids_tensor = torch.tensor(input_ids, device=device)
    input_ids, word_map = build_word_map(text)
    if input_ids is None:
        print(f"  word_map failed for idx {idx}")
    for method in methods:
        try:
            if method == "lime":
                token_attr = get_lime_attr(text, label, input_ids)
            elif method == "shap":
                token_attr = get_shap_attr(text, label, input_ids)
            else:
                token_attr = get_ig_attr(input_ids, label)

            if token_attr is None or np.allclose(token_attr, 0):
                invalid_counts[method] += 1
                pbar.update(1)
                continue

            word_attr = token_to_word_attr(token_attr, word_map)
            K = min(TOP_K, len(word_attr))

            # COMPREHENSIVENESS
            comp_rank = np.argsort(-word_attr)[:K]
            comp_curve, _ = compute_curve(input_ids_tensor, word_map, comp_rank, label)
            comp = comp_curve.mean()

            # SUFFICIENCY (correct)
            suff = compute_sufficiency(input_ids_tensor, word_map, comp_rank, label)

            # NAOPC
            upper, lower = compute_limits(input_ids_tensor, word_map, word_attr, label)
            naopc = (comp - lower) / (upper - lower + 1e-8)
            naopc = np.clip(naopc, 0.0, 1.0)

            results_buffer.append({
                "id": idx,
                "method": method,
                "comp": comp,
                "suff": suff,
                "naopc": naopc
            })

            valid_counts[method] += 1

        except Exception:
            invalid_counts[method] += 1

        pbar.update(1)

    if len(results_buffer) >= BUFFER_SIZE * len(methods):
        pd.DataFrame(results_buffer).to_csv(output_path, mode="a", header=False, index=False)
        results_buffer = []
        gc.collect()

if results_buffer:
    pd.DataFrame(results_buffer).to_csv(output_path, mode="a", header=False, index=False)

pbar.close()

# ================= SUMMARY =================
res = pd.read_csv(output_path)

print("\n===== FINAL SUMMARY =====")
for m in methods:
    sub = res[res.method == m]
    print(f"\n{m.upper()}:")
    print(f"  valid: {valid_counts[m]}")
    print(f"  invalid: {invalid_counts[m]}")
    print(f"  comp: {sub.comp.mean():.4f}")
    print(f"  suff: {sub.suff.mean():.4f}")
    print(f"  naopc: {sub.naopc.mean():.4f}")






###########  RANK AGREEMENT #################
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def rank_agreement_for_example(word_attr1, word_attr2, k):
    top1 = np.argsort(-np.abs(word_attr1))[:k]  #explanation 1 
    top2 = np.argsort(-np.abs(word_attr2))[:k]  #explanation 2 
    matches = sum(1 for i in range(k) if top1[i] == top2[i]) #comparation
    print(f"ra: ", matches/k)
    return matches / k

if os.path.exists(output_path):
    existing = pd.read_csv(output_path)
    processed_ids = set(existing['id'].tolist())
    write_header = False
    print(f"processed:: {len(processed_ids)}")
else:
    processed_ids = set()
    write_header = True
    pd.DataFrame(columns=['id', 'method', 'rank_agreement']).to_csv(output_path, index=False)

methods = ['lime', 'shap', 'ig']
buffer = []


pbar = tqdm(total=len(df), desc="Processing texts")
for idx in df.index:
    if idx in processed_ids:
        pbar.update(1)
        continue

    text = df.loc[idx, 'textNdesc']
    label = int(df.loc[idx, 'label'])
    input_ids, word_map = build_word_map(text)
    if input_ids is None:
        print(f"word_map failed for idx {idx}")
        pbar.update(1)
        continue

    input_ids_tensor = torch.tensor(input_ids, device=device)

    for method in methods:
        if method == 'lime':
            print("LIME: ")
            attr1 = get_lime_attr(text, label, input_ids)
            attr2 = get_lime_attr(text, label, input_ids)   
        elif method == 'shap':
            print("SHAP: ")
            attr1 = get_shap_attr(text, label, input_ids)
            attr2 = get_shap_attr(text, label, input_ids)
        else:  # ig
            print("IG: ")
            attr1 = get_ig_attr(input_ids, label)
            attr2 = get_ig_attr(input_ids, label)

        if attr1 is None or attr2 is None:
            continue

        word_attr1 = token_to_word_attr(attr1, word_map)
        word_attr2 = token_to_word_attr(attr2, word_map)
        if len(word_attr1) == 0 or len(word_attr2) == 0:
            continue

        ra = rank_agreement_for_example(word_attr1, word_attr2, TOP_K)
        buffer.append({
            'id': idx,
            'method': method,
            'rank_agreement': ra
        })

    if len(buffer) >= 10 or idx == df.index[-1]:
        pd.DataFrame(buffer).to_csv(output_path, mode='a', header=False, index=False)
        buffer = []
        gc.collect()

    pbar.update(1)

pbar.close()

# ==================== SUMMARY ====================
results = pd.read_csv(output_path)
print("\n-------RANK AGREEMENT (RA) SUMMARY-------")
for m in methods:
    sub = results[results.method == m]
    print(f"{m.upper()}:")
    print(f"  Valid examples: {len(sub)}")
    print(f"  Mean RA (top-{TOP_K}): {sub['rank_agreement'].mean():.4f}")
    print(f"  Std: {sub['rank_agreement'].std():.4f}")
