#Google Colab

#pip install shap
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import shap
import torch

from google.colab import drive
drive.mount('/content/drive')

#-----------MODEL LOAD--------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "/content/drive/MyDrive/FHM/bert_full_250126"
model = BertForSequenceClassification.from_pretrained(save_path)
tokenizer = BertTokenizer.from_pretrained(save_path)
model.to(device)
model.eval()

sample_df=pd.read_csv("yourpath/test_sample_350_stratified.csv")


#---------------GET PREDICTIONS---------
def predict_proba(texts):
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    texts = [str(t) for t in texts]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()


#---------------EXPLANATION-------------
masker = shap.maskers.Text(tokenizer, mask_token=tokenizer.mask_token)
shap_explainer = shap.Explainer(predict_proba, masker, output_names=["non-toxic", "toxic"])

#--------------DISTRIBUTION-----------
def get_shap_distribution(text, target_class):
    """
    returns SHAP probability distribution
    """
    # shap_values: list for every class, shape (len(text_tokens),)
    shap_values = shap_explainer([text])
    #get target class
    shap_target = shap_values.values[0][:, target_class]
    shap_other = shap_values.values[0][:, 1 - target_class]
    all_features = list(range(len(shap_target)))

    P = np.abs(np.array(shap_target, dtype=float))
    Q = np.abs(np.array(shap_other, dtype=float))

    return P, Q

# ================= KL =================
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

#-------------MAIN FUNCTION------------
def calculate_contrastivity_shap(text):
    try:
        probs = predict_proba([text])[0]
        target = int(np.argmax(probs))

        P, Q = get_shap_distribution(text, target)
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

#-------------ALL DATA LOOP------------
def run_contrastivity_shap(df, text_col="textNdesc", output_path="yourpath", resume=True):

    if resume and os.path.exists(output_path):
        existing = pd.read_csv(output_path, on_bad_lines='skip')
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
        res = calculate_contrastivity_shap(text)
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

    df_res = pd.read_csv(output_path, on_bad_lines='skip')
    valid = df_res[df_res.success == True]

    print("\n===== SUMMARY =====")
    print(f"valid: {len(valid)} / {len(df_res)}")
    print(f"KL_sym: mean={valid.kl_sym.mean():.4f}, std={valid.kl_sym.std():.4f}")
    print(f"KL_norm: mean={valid.kl_norm.mean():.4f}, std={valid.kl_norm.std():.4f}")

    return df_res

results_df = run_contrastivity_shap(df=sample_df, text_col="textNdesc", output_path="yourpath", resume=True)

