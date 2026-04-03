import cv2
import csv
import os
from datetime import datetime

# ─── importing deepface 
try:
    from deepface import DeepFace
except ImportError:
    print("DeepFace not found. Run:  pip install deepface")
    exit(1)

# ───  setting up
LOG_FILE   = "emotion_log.csv"
EMOTIONS   = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FONT       = cv2.FONT_HERSHEY_SIMPLEX

# Colour per emotion  (B, G, R)
COLOURS = {
    "happy":    (0,   220,  80),
    "sad":      (220,  80,   0),
    "angry":    (0,    30, 220),
    "fear":     (150,   0, 200),
    "disgust":  (0,   150,  60),
    "surprise": (0,   200, 220),
    "neutral":  (180, 180, 180),
}

# ─── CSV logger
def init_csv(path: str) -> None:
    """Create the log file with a header row if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "emotion", "confidence_%"])
        print(f"[INFO] Log file created: {path}")


def log_emotion(path: str, emotion: str, confidence: float) -> None:
    
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, emotion, f"{confidence:.1f}"])


# ─── Face + emotion detection 
def analyse_frame(frame):
    
    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,   
            silent=True,
        )
        if not isinstance(results, list):
            results = [results]
        return results
    except Exception:
        return []


# ─── drawing overlay 
def draw_overlay(frame, results):
    """Draw bounding box + emotion label on the frame."""
    for r in results:
        region  = r.get("region", {})
        x, y    = region.get("x", 0),  region.get("y", 0)
        w, h    = region.get("w", 0),  region.get("h", 0)

        emotions    = r.get("emotion", {})
        top_emotion = max(emotions, key=emotions.get)
        confidence  = emotions[top_emotion]

        colour  = COLOURS.get(top_emotion, (255, 255, 255))
        label   = f"{top_emotion.upper()}  {confidence:.0f}%"

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

       
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), colour, -1)

        
        cv2.putText(frame, label, (x + 5, y - 6),
                    FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return frame


# ─── Accuracy summary from log
def print_accuracy_summary(path: str) -> None:
    """Print a basic count of each emotion from the saved log."""
    if not os.path.exists(path):
        print("[INFO] No log file found.")
        return

    counts = {e: 0 for e in EMOTIONS}
    total  = 0

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            em = row.get("emotion", "").lower()
            if em in counts:
                counts[em] += 1
                total += 1

    if total == 0:
        print("[INFO] Log is empty — nothing to summarise.")
        return

    print("\n" + "═" * 40)
    print("  EMOTION DETECTION SUMMARY")
    print("═" * 40)
    for emotion, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct  = (count / total) * 100
        bar  = "█" * int(pct / 4)
        print(f"  {emotion:<10}  {bar:<25}  {pct:5.1f}%  ({count})")
    print(f"\n  Total detections: {total}")
    print("═" * 40 + "\n")


# main loop
def main():
    init_csv(LOG_FILE)

    cap = cv2.VideoCapture(0)         
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        return

    print("[INFO] Webcam opened. Press  Q  to quit.")

    frame_count  = 0
    analyse_every = 5                 

    last_results  = []                 
    last_logged   = ""                

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Empty frame received — retrying…")
            continue

        frame_count += 1

        # Only run the (slow) DeepFace call every few frames
        if frame_count % analyse_every == 0:
            last_results = analyse_frame(frame)

            # Log only when top emotion changes
            for r in last_results:
                emotions    = r.get("emotion", {})
                if not emotions:
                    continue
                top_emotion = max(emotions, key=emotions.get)
                confidence  = emotions[top_emotion]
                entry_key   = f"{top_emotion}-{confidence:.0f}"
                if entry_key != last_logged:
                    log_emotion(LOG_FILE, top_emotion, confidence)
                    last_logged = entry_key

        
        output = draw_overlay(frame.copy(), last_results)

        
        cv2.putText(output, "Press Q to quit", (10, 28),
                    FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        cv2.imshow("Real-Time Emotion Detection", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam released.")

    print_accuracy_summary(LOG_FILE)



if __name__ == "__main__":
    main()
