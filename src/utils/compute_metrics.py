import numpy as np
from iouEval import iouEval
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(eval_preds):
    metric = dict()

    # 예측값과 레이블 추출
    logits = eval_preds.predictions  # shape: [B, num_classes]
    labels = eval_preds.label_ids    # shape: [B]

    preds = np.argmax(logits, axis=1)  # shape: [B]

    metric["accuracy"] = accuracy_score(labels, preds)
    metric["f1_score"] = f1_score(labels, preds, average="macro")
    metric["precision"] = precision_score(labels, preds, average="macro", zero_division=0)
    metric["recall"] = recall_score(labels, preds, average="macro", zero_division=0)

    """
    print("===== Validation Example ===============")
    sample_idx = np.random.randint(0, len(labels), 5)
    for idx in sample_idx:
        print(f"sample {idx}")
        print(f"answer: {labels[idx]}")
        print(f"pred: {preds[idx]}\n")
    print("=======================================")
    """
    return metric