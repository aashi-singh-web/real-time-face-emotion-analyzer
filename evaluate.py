import csv
import os
import argparse
from collections import Counter
from datetime import datetime

LOG_FILE = "emotion_log.csv"
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_log(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def print_bar(label: str, value: float, total: float, width: int = 30):
    filled = int((value / total) * width) if total > 0 else 0
    bar    = "█" * filled + "░" * (width - filled)
    print(f"  {label:<12} {bar}  {value / total * 100:5.1f}%  ({int(value)})")


def summarise(rows: list[dict]) -> None:
    total = len(rows)
    if total == 0:
        print("[INFO] No data to summarise.")
        return

    counts = Counter(r["emotion"].lower() for r in rows if "emotion" in r)
    confs  = {}
    for r in rows:
        em = r.get("emotion", "").lower()
        try:
            c = float(r.get("confidence_%", 0))
        except ValueError:
            c = 0
        confs.setdefault(em, []).append(c)

    print("\n" + "═" * 58)
    print("   EMOTION DETECTION LOG SUMMARY")
    print("═" * 58)
    print(f"   Total detections : {total}")

    # Time range
    try:
        times = [datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S") for r in rows]
        span  = (max(times) - min(times)).seconds
        print(f"   Session duration : {span // 60}m {span % 60}s")
        print(f"   Start            : {min(times)}")
        print(f"   End              : {max(times)}")
    except Exception:
        pass

    print("\n   FREQUENCY & AVERAGE CONFIDENCE")
    print("   " + "-" * 55)
    for em in sorted(counts, key=lambda e: -counts[e]):
        avg_conf = sum(confs.get(em, [0])) / max(len(confs.get(em, [1])), 1)
        print_bar(em, counts[em], total)
        print(f"   {'':12} Avg confidence: {avg_conf:.1f}%")
    print("═" * 58 + "\n")


def compare_with_ground_truth(log_rows: list[dict], gt_path: str) -> None:
    """
    Ground truth CSV format:
        timestamp, true_emotion
    Matches log rows by timestamp (exact).
    """
    if not os.path.exists(gt_path):
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        return

    with open(gt_path, newline="") as f:
        gt_rows = {r["timestamp"]: r["true_emotion"].lower()
                   for r in csv.DictReader(f)}

    correct = 0
    total   = 0
    confusion: dict[str, Counter] = {}

    for r in log_rows:
        ts       = r.get("timestamp", "")
        pred     = r.get("emotion", "").lower()
        true_em  = gt_rows.get(ts)
        if true_em is None:
            continue
        total += 1
        confusion.setdefault(true_em, Counter())[pred] += 1
        if pred == true_em:
            correct += 1

    if total == 0:
        print("[INFO] No matching timestamps between log and ground truth.")
        return

    accuracy = correct / total * 100
    print("\n" + "═" * 40)
    print(f"   ACCURACY  :  {accuracy:.1f}%  ({correct}/{total})")
    print("═" * 40)
    print("\n   CONFUSION MATRIX (true → predicted)")
    # header = f"   {'True \\ Pred':<12}" + "".join(f"{e[:7]:>8}" for e in EMOTIONS)
    header = "   {:<12}".format("True \\ Pred") + "".join(f"{e[:7]:>8}" for e in EMOTIONS)
    print(header)
    print("   " + "-" * (len(header) - 3))
    for true_em in EMOTIONS:
        row = f"   {true_em:<12}"
        row += "".join(f"{confusion.get(true_em, Counter()).get(pred, 0):>8}"
                       for pred in EMOTIONS)
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(description="Emotion log analyser")
    parser.add_argument("--log",          default=LOG_FILE, help="Path to emotion_log.csv")
    parser.add_argument("--ground-truth", default=None,     help="Path to ground truth CSV")
    args = parser.parse_args()

    rows = load_log(args.log)
    summarise(rows)

    if args.ground_truth:
        compare_with_ground_truth(rows, args.ground_truth)


if __name__ == "__main__":
    main()
