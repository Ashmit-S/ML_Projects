# Vision-Based Safety Compliance Monitoring

Link to Dataset Used: https://universe.roboflow.com/nexplore-r2tbg/ppe-bhtl9/dataset/2

This project implements a clean computer vision pipeline for safety compliance monitoring using:

- YOLOv8 (Ultralytics)
- ByteTrack (through `YOLO.track(..., persist=True)`)
- Temporal logic for robust violation decisions

The goal is to show that frame-by-frame detections are noisy, and temporal consistency improves reliability.

## Pipeline

1. Detection
- YOLO detects objects in each frame (person, helmet/hard_hat, vest if available in the model labels).

2. Tracking
- Built-in ByteTrack keeps a persistent track ID per person across frames.

3. Temporal Consistency Rule
- For each tracked person (`track_id`), the system keeps a state record.
- If helmet is missing for N consecutive frames (default: 10), the person is flagged as a violation.
- If helmet reappears, the counter resets and the violation state clears.
- If model labels include explicit missing-helmet classes (for example `no_hard_hat`), those detections are also used as negative evidence.

4. Output
- Annotated bounding boxes with track IDs.
- "VIOLATION" text overlays for non-compliant workers.
- Violation messages printed in console.
- Output video saved to disk.

## Project Files

- `main.py`: End-to-end pipeline with modular functions for class resolution, detection extraction, rule logic, and rendering.
- `train_eval.py`: Fine-tune YOLOv8 on selected PPE classes and evaluate on held-out split with mAP@0.5, precision, and recall.
- `requirements.txt`: Python dependencies.
- `ppe.v3i.yolov8/`: Dataset metadata and labels available in workspace context.


## Edge Cases Handled

- Missing track IDs: boxes are still drawn with `ID:NA`, but no temporal state is assigned.
- People entering/exiting frame: stale tracks are pruned after configurable age.
- Frame drops/read failures: tolerates temporary read failures before stopping.

## Key Temporal Logic Idea

A single missed helmet detection in one frame should not immediately trigger a violation. The consecutive-frame rule suppresses transient detector noise and makes alerts more stable in practical deployments.
