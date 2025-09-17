import json, os
from config import DATASET_NAME, PREDICTIONS_DIR, MODEL, ROOT

IN_PATH = f"{PREDICTIONS_DIR}/{DATASET_NAME}_{MODEL}_predictions.json"
OUT_PATH = f"{ROOT}/{DATASET_NAME}/{DATASET_NAME}_per_class_accuracy.json"

def pct(x, n): return (100.0 * x / n) if n else 0.0


if __name__ == "__main__":

    with open(IN_PATH, "r") as f:
        preds = json.load(f)

    stats = {0: {"total": 0, "correct": 0}, 1: {"total": 0, "correct": 0}}

    for _, rec in preds.items():
        y_true = int(rec["true_label"])
        y_pred = int(rec.get("pred_label", rec.get("prediction", y_true)))
        correct = bool(rec.get("correct", y_true == y_pred))
        if y_true in (0, 1):
            stats[y_true]["total"] += 1
            stats[y_true]["correct"] += int(correct)

    for cls in (0, 1):
        t = stats[cls]["total"]
        c = stats[cls]["correct"]
        print(f"class {cls}: {c}/{t} correct ({pct(c, t):.2f}%)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out = {
        "dataset": DATASET_NAME,
        "class_0": {
            "total": stats[0]["total"],
            "correct": stats[0]["correct"],
            "accuracy_pct": pct(stats[0]["correct"], stats[0]["total"]),
        },
        "class_1": {
            "total": stats[1]["total"],
            "correct": stats[1]["correct"],
            "accuracy_pct": pct(stats[1]["correct"], stats[1]["total"]),
        },
    }

    print(out)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")
