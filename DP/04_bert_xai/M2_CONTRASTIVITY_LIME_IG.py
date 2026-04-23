#Google Colab

#pip install lime
#pip install captum

import os
import gc
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from lime.lime_text import LimeTextExplainer
from captum.attr import IntegratedGradients
from scipy.special import logit
from scipy.stats import pearsonr

drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------MODEL LOAD--------------------
save_path = "/content/drive/MyDrive/FHM/bert_full_250126"
model = BertForSequenceClassification.from_pretrained(save_path)
tokenizer = BertTokenizer.from_pretrained(save_path)
model.to(device)
model.eval()

#------------------DATA LOAD------------------------------
sample_df=pd.read_csv("yourpath/test_sample_350_stratified.csv")
class_names = ["non-toxic", "toxic"]


#---------------CONFIG------------
NUM_FEATURES = 10
NUM_SAMPLES = 500

#---------------EXPLANATIONS----------
lime_explainer = LimeTextExplainer(class_names=["non-toxic", "toxic"], split_expression=r"\W+", bow=False)


#---------------PREDICTIONS--------
def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()


# ================= ATTR (UNIFIED) =================
def get_lime_distribution(text, target_class):
    """
    return lime probability distribution
    """
    exp = lime_explainer.explain_instance(text_instance=text, classifier_fn=predict_proba, num_features=NUM_FEATURES, num_samples=NUM_SAMPLES)

    #weights for both classes
    weights_target = dict(exp.local_exp.get(target_class, []))
    weights_other = dict(exp.local_exp.get(1 - target_class, []))
    all_features = sorted(set(weights_target) | set(weights_other))

    P = np.array([abs(weights_target.get(f, 0.0)) for f in all_features])
    Q = np.array([abs(weights_other.get(f, 0.0)) for f in all_features])

    return P, Q


#---------CALCULATE KL-----------
def compute_kl(P, Q):
    eps = 1e-10

    if P.sum() == 0:
        P = np.ones_like(P) / len(P)
    else:
        P = P / P.sum()

    if Q.sum() == 0:
        Q = np.ones_like(Q) / len(Q)
    else:
        Q = Q / Q.sum()

    P = P + eps
    Q = Q + eps

    P = P / P.sum()
    Q = Q / Q.sum()

    kl_pq = np.sum(P * np.log(P / Q))
    kl_qp = np.sum(Q * np.log(Q / P))

    return kl_pq, kl_qp


#-----------MAIN FUNC-------------
def calculate_contrastivity_lime(text):
    try:
        probs = predict_proba([text])[0]
        target = int(np.argmax(probs))

        P, Q = get_lime_distribution(text, target)

        kl_pq, kl_qp = compute_kl(P, Q)

        kl_sym = 0.5 * (kl_pq + kl_qp)
        kl_norm = kl_sym / (kl_sym + 1)

        return {
            "kl_pq": kl_pq,
            "kl_qp": kl_qp,
            "kl_sym": kl_sym,
            "kl_norm": kl_norm,
            "target": target,
            "prob_0": float(probs[0]),
            "prob_1": float(probs[1]),
            "n_features": len(P),
            "success": True
        }

    except Exception as e:
        return {
            "kl_pq": None,
            "kl_qp": None,
            "kl_sym": None,
            "kl_norm": None,
            "target": None,
            "prob_0": None,
            "prob_1": None,
            "n_features": None,
            "success": False,
            "error": str(e)
        }


#--------------RUN ALL DATA----------
def run_contrastivity_lime(
    df, text_col="textNdesc", output_path="yourpath", resume=True):

    if resume and os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        processed = set(existing["index"])
        write_header = False
        print(f"[resume] {len(processed)} processed")
    else:
        processed = set()
        write_header = True

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if idx in processed:
            continue
        text = str(row[text_col])
        res = calculate_contrastivity_lime(text)
        res["index"] = idx
        res["text"] = text

        pd.DataFrame([res]).to_csv(
            output_path,
            mode="a",
            header=write_header,
            index=False
        )

        write_header = False
        gc.collect()
    print("\nDONE")

    df_res = pd.read_csv(output_path)
    valid = df_res[df_res.success == True]

    print("\n===== SUMMARY =====")
    print(f"valid: {len(valid)} / {len(df_res)}")
    print(f"KL_sym: mean={valid.kl_sym.mean():.4f}, std={valid.kl_sym.std():.4f}")
    print(f"KL_norm: mean={valid.kl_norm.mean():.4f}, std={valid.kl_norm.std():.4f}")

    return df_res

results_df = run_contrastivity_lime(df=sample_df, text_col="textNdesc", output_path="yourpath", resume=True)
