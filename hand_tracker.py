import argparse
import math
import time
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

PALM_WIDTH_CM = 8.0
DEFAULT_FOCAL_PX = 600.0

TIPS = [4, 8, 12, 16, 20]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def landmarks_to_px(hand_landmarks, w, h):
    return [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]


def finger_states(pts):
    """Return [thumb, index, middle, ring, pinky] booleans (True = extended).
    Uses distances only, so it works for any hand rotation."""
    wrist = pts[0]
    size = dist(pts[0], pts[9])  # wrist -> middle MCP, used for scale

    # Thumb: tip must be far from the index MCP
    thumb = dist(pts[4], pts[5]) / size > 0.55

    states = [thumb]
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        states.append(dist(wrist, pts[tip]) > dist(wrist, pts[pip]) * 1.1)
    return states, size


def classify_gesture(pts):
    states, size = finger_states(pts)
    thumb, index, middle, ring, pinky = states
    pinch = dist(pts[4], pts[8]) / size

    if pinch < 0.25 and middle and ring and pinky:
        return "OK"
    if all(states):
        return "Open Palm"
    if not any(states):
        return "Fist"

    # Thumb only -> up / down based on vertical direction
    if thumb and not (index or middle or ring or pinky):
        dy = pts[0][1] - pts[4][1]  # positive = thumb tip above wrist
        if dy > 0.6 * size:
            return "Thumbs Up"
        if dy < -0.6 * size:
            return "Thumbs Down"
        return "Thumb Out"

    if index and middle and ring and pinky and not thumb:
        return "Four"
    if index and middle and ring and not pinky:
        return "Three"
    if index and pinky and not middle and not ring:
        return "Rock"
    if index and middle and not ring and not pinky:
        return "Peace"
    if index and not (middle or ring or pinky):
        return "Pointing"
    return "Unknown"


class Smoother:
    """Exponential moving average to reduce jitter in measurements."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


def put_label(img, text, org, color=(255, 255, 255), scale=0.6, bg=(0, 0, 0)):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + base), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--calib-dist", type=float, default=40.0, help="cm, used by the 'c' calibration key")
    ap.add_argument("--max-hands", type=int, default=2)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    focal = DEFAULT_FOCAL_PX
    mirror = True
    smoothers = {}  # per-hand smoothing: key -> (pinch, cam_dist)
    prev_time = time.time()
    fps = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)

            top_gesture = None
            calib_palm_px = None

            if result.multi_hand_landmarks:
                for i, (hl, handed) in enumerate(
                    zip(result.multi_hand_landmarks, result.multi_handedness)
                ):
                    pts = landmarks_to_px(hl, w, h)
                    label = handed.classification[0].label  # "Left"/"Right"

                    mp_draw.draw_landmarks(
                        frame,
                        hl,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    # --- Distance measurement ---
                    palm_px = dist(pts[5], pts[17])
                    calib_palm_px = palm_px
                    cm_per_px = PALM_WIDTH_CM / palm_px if palm_px > 1 else 0
                    cam_cm = (PALM_WIDTH_CM * focal) / palm_px if palm_px > 1 else 0
                    pinch_cm = dist(pts[4], pts[8]) * cm_per_px

                    sm = smoothers.setdefault(i, (Smoother(), Smoother()))
                    pinch_cm = sm[0].update(pinch_cm)
                    cam_cm = sm[1].update(cam_cm)

                    # Line between thumb tip and index tip
                    t = (int(pts[4][0]), int(pts[4][1]))
                    ix = (int(pts[8][0]), int(pts[8][1]))
                    mid = ((t[0] + ix[0]) // 2, (t[1] + ix[1]) // 2)
                    cv2.line(frame, t, ix, (0, 255, 255), 2)
                    cv2.circle(frame, t, 8, (0, 255, 255), -1)
                    cv2.circle(frame, ix, 8, (0, 255, 255), -1)
                    put_label(frame, f"{pinch_cm:.1f} cm", (mid[0] + 10, mid[1]), (0, 255, 255), 0.55)

                    # --- Gesture ---
                    gesture = classify_gesture(pts)
                    if top_gesture is None:
                        top_gesture = gesture

                    # Bounding box + info
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, y1 = int(min(xs)) - 15, int(min(ys)) - 15
                    x2, y2 = int(max(xs)) + 15, int(max(ys)) + 15
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    put_label(frame, f"{label}: {gesture}", (x1, max(y1 - 8, 20)), (0, 255, 0))
                    put_label(frame, f"Dist: {cam_cm:.0f} cm", (x1, min(y2 + 22, h - 5)), (255, 200, 0))

            # --- HUD ---
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            cv2.putText(
                frame,
                f"Gesture: {top_gesture or 'None'}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            put_label(frame, f"FPS: {fps:.0f}", (10, h - 12), (255, 255, 255), 0.5)
            put_label(frame, "c: calibrate  m: mirror  q: quit", (w - 285, h - 12), (200, 200, 200), 0.5)

            cv2.imshow("Hand Tracking - Distance - Gestures", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("c") and calib_palm_px:
                focal = (calib_palm_px * args.calib_dist) / PALM_WIDTH_CM
                print(f"Calibrated focal length: {focal:.1f}px at {args.calib_dist} cm")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
