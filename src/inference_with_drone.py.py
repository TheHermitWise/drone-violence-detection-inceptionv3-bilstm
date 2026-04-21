"""
Refactored Tello Violence Monitor
Correct probability logic + threshold handling
Compatible with CNN (InceptionV3) + BiLSTM trained model
"""

from djitellopy import Tello
import cv2
import time
import os
import argparse
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


# =========================================================
# CONFIG
# =========================================================
FRAMES_PER_SEQUENCE = 20


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="violence_detection_cnn_bilstm.h5")
    p.add_argument("--img-size", default="299,299")
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--ema-alpha", type=float, default=0.4)
    p.add_argument("--debounce-frames", type=int, default=3)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--show-ui", action="store_true", default=True)

    args = p.parse_args()
    w, h = args.img_size.split(",")
    args.img_size = (int(w), int(h))
    return args


# =========================================================
# LOGGING
# =========================================================
def setup_logging(outdir):
    os.makedirs(outdir, exist_ok=True)

    logger = logging.getLogger("violence")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(
        os.path.join(outdir, "run.log"),
        maxBytes=2_000_000,
        backupCount=3
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# =========================================================
# TEMPORAL SMOOTHER
# =========================================================
class Smoother:
    def __init__(self, alpha=0.4, debounce=3):
        self.alpha = alpha
        self.debounce = debounce
        self.ema = None
        self.stable_label = None
        self.counter = 0

    def update(self, raw_label, p_viol):
        # Smooth only violence probability
        if self.ema is None:
            self.ema = p_viol
        else:
            self.ema = self.alpha * p_viol + (1 - self.alpha) * self.ema

        # Debounce label switching
        if self.stable_label is None:
            self.stable_label = raw_label

        if raw_label != self.stable_label:
            self.counter += 1
            if self.counter >= self.debounce:
                self.stable_label = raw_label
                self.counter = 0
        else:
            self.counter = 0

        return self.stable_label, self.ema


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()
    logger = setup_logging(args.outdir)

    logger.info("Loading BiLSTM model...")
    model = load_model(args.model)

    logger.info("Loading CNN feature extractor...")
    cnn = InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )
    cnn.trainable = False

    logger.info(f"Feature dimension: {cnn.output_shape[-1]}")

    feature_buffer = deque(maxlen=FRAMES_PER_SEQUENCE)
    smoother = Smoother(args.ema_alpha, args.debounce_frames)

    # Connect to Tello
    tello = Tello()
    tello.connect()
    logger.info(f"Battery: {tello.get_battery()}%")

    tello.streamon()
    frame_read = tello.get_frame_read()

    fps_times = deque(maxlen=30)

    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                continue

            t0 = time.perf_counter()

            # =====================================================
            # PREPROCESS
            # =====================================================
            resized = cv2.resize(frame, args.img_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            arr = np.expand_dims(rgb.astype(np.float32), axis=0)
            arr = preprocess_input(arr)

            # =====================================================
            # FEATURE EXTRACTION
            # =====================================================
            features = cnn.predict(arr, verbose=0)[0]
            feature_buffer.append(features)

            label = "Waiting..."
            p_viol = 0.0
            p_nonv = 0.0

            # =====================================================
            # SEQUENCE PREDICTION
            # =====================================================
            if len(feature_buffer) == FRAMES_PER_SEQUENCE:
                sequence = np.expand_dims(np.array(feature_buffer), axis=0)
                prediction = model.predict(sequence, verbose=0)[0]

                p_viol = float(prediction[0])
                p_nonv = float(prediction[1])

                # Argmax decision
                raw_label = "Violência" if p_viol >= p_nonv else "Sem Violência"

                # Apply threshold only to violence
                if p_viol >= p_nonv and p_viol >= args.threshold:
                    label = "Violência"
                else:
                    label = "Sem Violência"

            # =====================================================
            # SMOOTHING
            # =====================================================
            stable_label, ema_pv = smoother.update(label, p_viol)

            latency_ms = (time.perf_counter() - t0) * 1000

            fps_times.append(time.perf_counter())
            if len(fps_times) >= 2:
                dt = fps_times[-1] - fps_times[0]
                fps = (len(fps_times) - 1) / dt
            else:
                fps = 0

            # =====================================================
            # DEBUG LOGGING
            # =====================================================
            logger.info(
                f"pV={p_viol:.3f} pNV={p_nonv:.3f} EMA_pV={ema_pv:.3f} "
                f"Label={stable_label}"
            )

            # =====================================================
            # UI
            # =====================================================
            if args.show_ui:
                color = (0, 0, 255) if stable_label == "Violência" else (0, 255, 0)

                cv2.putText(frame,
                            f"{stable_label} | pV={p_viol:.2f} pNV={p_nonv:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2)

                cv2.putText(frame,
                            f"EMA pV: {ema_pv:.2f}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2)

                cv2.putText(frame,
                            f"FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2)

                cv2.imshow("Tello Violence Monitor", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        tello.streamoff()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()