from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
from ultralytics import YOLO


BBox = Tuple[float, float, float, float]


@dataclass
class PersonState:
    missing_helmet_count: int = 0
    helmet_violation_active: bool = False
    missing_vest_count: int = 0
    vest_violation_active: bool = False
    last_seen_frame: int = -1

    @property
    def violation_active(self) -> bool:
        return self.helmet_violation_active or self.vest_violation_active


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vision-based safety compliance monitoring with YOLOv8 + ByteTrack + temporal logic"
    )
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", default="output_annotated.mp4", help="Path to output video file")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to YOLO model weights (yolov8n.pt or custom PPE model)",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Tracker config name/path passed to YOLO.track",
    )
    parser.add_argument(
        "--helmet-missing-frames",
        type=int,
        default=30,
        help="Consecutive frames without helmet before flagging violation",
    )
    parser.add_argument(
        "--vest-missing-frames",
        type=int,
        default=30,
        help="Consecutive frames without vest before flagging violation",
    )
    parser.add_argument(
        "--helmet-overlap-threshold",
        type=float,
        default=0.1,
        help="Minimum overlap score to associate helmet to a person",
    )
    parser.add_argument(
        "--vest-overlap-threshold",
        type=float,
        default=0.15,
        help="Minimum overlap score to associate vest to a person",
    )
    parser.add_argument(
        "--max-track-age",
        type=int,
        default=30,
        help="Remove track state if person not seen for this many frames",
    )
    parser.add_argument(
        "--max-consecutive-frame-drops",
        type=int,
        default=5,
        help="Tolerance for consecutive failed frame reads before stopping",
    )
    parser.add_argument(
        "--person-class",
        default="person",
        help="Person class name in model labels",
    )
    parser.add_argument(
        "--helmet-classes",
        default="helmet,hard_hat",
        help="Comma-separated helmet-like class names",
    )
    parser.add_argument(
        "--vest-classes",
        default="vest,safety_vest",
        help="Comma-separated vest-like class names",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live annotated preview window",
    )
    return parser.parse_args()


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _split_names(value: str) -> Set[str]:
    return {_normalize_name(token) for token in value.split(",") if token.strip()}


def resolve_class_ids(model: YOLO, class_names: Iterable[str]) -> Set[int]:
    names = getattr(model.model, "names", {}) or {}
    requested = {_normalize_name(name) for name in class_names}

    matched: Set[int] = set()
    for class_id, class_name in names.items():
        if _normalize_name(str(class_name)) in requested:
            matched.add(int(class_id))
    return matched


def bbox_intersection_ratio(inner: BBox, outer: BBox) -> float:
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    inner_area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter_area / inner_area


def is_helmet_on_person(person_box: BBox, helmet_boxes: Sequence[BBox], overlap_threshold: float) -> bool:
    px1, py1, px2, py2 = person_box
    height = max(1.0, py2 - py1)

    # Helmet is expected near the top area of a person's box.
    person_head_region: BBox = (px1, py1, px2, py1 + 0.4 * height)

    for helmet_box in helmet_boxes:
        overlap = bbox_intersection_ratio(helmet_box, person_head_region)
        if overlap >= overlap_threshold:
            return True
    return False


def is_vest_on_person(person_box: BBox, vest_boxes: Sequence[BBox], overlap_threshold: float) -> bool:
    px1, py1, px2, py2 = person_box
    height = max(1.0, py2 - py1)

    # Vest is expected on the torso: roughly 20%-80% of the person's bounding box height.
    person_torso_region: BBox = (px1, py1 + 0.2 * height, px2, py1 + 0.8 * height)

    for vest_box in vest_boxes:
        overlap = bbox_intersection_ratio(vest_box, person_torso_region)
        if overlap >= overlap_threshold:
            return True
    return False


def draw_label(frame, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    pad = 4
    x1, y1 = x, y - th - (2 * pad)
    x2, y2 = x + tw + (2 * pad), y
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(frame, text, (x + pad, y - pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def extract_detections(result) -> List[Dict[str, Optional[object]]]:
    detections: List[Dict[str, Optional[object]]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(classes)

    for box, class_id, track_id in zip(xyxy, classes, track_ids):
        detections.append(
            {
                "bbox": (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                "class_id": int(class_id),
                "track_id": int(track_id) if track_id is not None else None,
            }
        )
    return detections


def cleanup_stale_states(states: Dict[int, PersonState], frame_index: int, max_track_age: int) -> None:
    stale_ids = [
        track_id
        for track_id, state in states.items()
        if frame_index - state.last_seen_frame > max_track_age
    ]
    for track_id in stale_ids:
        del states[track_id]


def run_pipeline(args: argparse.Namespace) -> None:
    model = YOLO(args.model)

    person_ids = resolve_class_ids(model, [args.person_class])
    helmet_ids = resolve_class_ids(model, _split_names(args.helmet_classes))
    vest_ids = resolve_class_ids(model, _split_names(args.vest_classes))

    if not person_ids:
        raise ValueError(
            "Could not find person class in model labels. "
            "Use --person-class with the label name used by your model."
        )

    print(f"Loaded model: {args.model}")
    print(f"Person class IDs: {sorted(person_ids)}")
    print(f"Helmet class IDs: {sorted(helmet_ids)}")
    print(f"Vest class IDs: {sorted(vest_ids)}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer: Optional[cv2.VideoWriter] = None
    frame_index = 0
    failed_reads = 0

    person_states: Dict[int, PersonState] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                is_early_failure = total_frames > 0 and current_pos < max(0, total_frames - 1)

                if is_early_failure and failed_reads < args.max_consecutive_frame_drops:
                    failed_reads += 1
                    print(
                        f"[WARN] Dropped/failed frame read at position {current_pos}. "
                        f"Consecutive drops: {failed_reads}"
                    )
                    continue
                break

            failed_reads = 0
            frame_index += 1

            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Unable to open output video for writing: {args.output}")

            # Built-in ByteTrack integration in Ultralytics via track(..., persist=True).
            results = model.track(
                source=frame,
                persist=True,
                tracker=args.tracker,
                verbose=False,
            )
            result = results[0]
            detections = extract_detections(result)

            helmet_boxes: List[BBox] = [
                det["bbox"]
                for det in detections
                if det["class_id"] in helmet_ids and det["bbox"] is not None
            ]
            vest_boxes: List[BBox] = [
                det["bbox"]
                for det in detections
                if det["class_id"] in vest_ids and det["bbox"] is not None
            ]

            active_violations = 0
            for det in detections:
                if det["class_id"] not in person_ids:
                    continue

                bbox = det["bbox"]
                if bbox is None:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                track_id = det["track_id"]

                if track_id is None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 2)
                    draw_label(frame, "ID:NA", x1, max(20, y1), (0, 200, 200))
                    continue

                state = person_states.setdefault(track_id, PersonState())
                state.last_seen_frame = frame_index

                helmet_on = is_helmet_on_person(bbox, helmet_boxes, args.helmet_overlap_threshold)
                vest_on = is_vest_on_person(bbox, vest_boxes, args.vest_overlap_threshold)

                # Helmet temporal logic
                if helmet_on:
                    state.missing_helmet_count = 0
                    state.helmet_violation_active = False
                else:
                    state.missing_helmet_count += 1
                    if state.missing_helmet_count >= args.helmet_missing_frames and not state.helmet_violation_active:
                        state.helmet_violation_active = True
                        print(
                            f"[VIOLATION] Frame {frame_index} | Track {track_id} | "
                            f"Missing helmet for {state.missing_helmet_count} consecutive frames"
                        )

                # Vest temporal logic
                if vest_on:
                    state.missing_vest_count = 0
                    state.vest_violation_active = False
                else:
                    state.missing_vest_count += 1
                    if state.missing_vest_count >= args.vest_missing_frames and not state.vest_violation_active:
                        state.vest_violation_active = True
                        print(
                            f"[VIOLATION] Frame {frame_index} | Track {track_id} | "
                            f"Missing vest for {state.missing_vest_count} consecutive frames"
                        )

                if state.violation_active:
                    active_violations += 1

                box_color = (0, 0, 255) if state.violation_active else (0, 180, 0)
                helmet_tag = "H:OK" if helmet_on else ("H:!" if state.helmet_violation_active else "H:NO")
                vest_tag = "V:OK" if vest_on else ("V:!" if state.vest_violation_active else "V:NO")
                if not vest_ids:
                    vest_tag = "V:NA"

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                draw_label(
                    frame,
                    f"ID:{track_id} {helmet_tag} {vest_tag}",
                    x1,
                    max(20, y1),
                    box_color,
                )

                if state.violation_active:
                    violation_parts = []
                    if state.helmet_violation_active:
                        violation_parts.append("NO HELMET")
                    if state.vest_violation_active:
                        violation_parts.append("NO VEST")
                    violation_text = " | ".join(violation_parts)
                    cv2.putText(
                        frame,
                        violation_text,
                        (x1, min(frame.shape[0] - 10, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            cleanup_stale_states(person_states, frame_index, args.max_track_age)

            if active_violations > 0:
                cv2.putText(
                    frame,
                    f"ACTIVE VIOLATIONS: {active_violations}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)

            if args.show:
                cv2.imshow("Safety Compliance Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Finished. Annotated output saved to: {args.output}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()