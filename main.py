from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import cv2

# Mediapipe imports matplotlib internally; set cache dir to writable workspace path.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

import mediapipe as mp

from analyzer import AnalysisResult, MusculoskeletalAnalyzer
from pose_utils import LandmarkSet, to_point

MP_POSE = mp.solutions.pose
MP_DRAW = mp.solutions.drawing_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="앉은 자세/보행 분석 기반 근골격계 위험도 추정 프로그램"
    )
    parser.add_argument("--mode", choices=["sitting", "gait"], default="sitting")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=120)
    return parser.parse_args()


def landmarks_from_result(result_pose_landmarks) -> Optional[LandmarkSet]:
    if result_pose_landmarks is None:
        return None

    lm = result_pose_landmarks.landmark
    idx = MP_POSE.PoseLandmark
    points = {
        "left_shoulder": to_point(lm[idx.LEFT_SHOULDER].x, lm[idx.LEFT_SHOULDER].y),
        "right_shoulder": to_point(lm[idx.RIGHT_SHOULDER].x, lm[idx.RIGHT_SHOULDER].y),
        "left_ear": to_point(lm[idx.LEFT_EAR].x, lm[idx.LEFT_EAR].y),
        "right_ear": to_point(lm[idx.RIGHT_EAR].x, lm[idx.RIGHT_EAR].y),
        "left_hip": to_point(lm[idx.LEFT_HIP].x, lm[idx.LEFT_HIP].y),
        "right_hip": to_point(lm[idx.RIGHT_HIP].x, lm[idx.RIGHT_HIP].y),
        "left_knee": to_point(lm[idx.LEFT_KNEE].x, lm[idx.LEFT_KNEE].y),
        "right_knee": to_point(lm[idx.RIGHT_KNEE].x, lm[idx.RIGHT_KNEE].y),
        "left_ankle": to_point(lm[idx.LEFT_ANKLE].x, lm[idx.LEFT_ANKLE].y),
        "right_ankle": to_point(lm[idx.RIGHT_ANKLE].x, lm[idx.RIGHT_ANKLE].y),
    }
    return LandmarkSet(points=points)


def draw_overlay(frame, mode: str, result: Optional[AnalysisResult]) -> None:
    title = "Sitting Analysis" if mode == "sitting" else "Gait Analysis"
    cv2.putText(frame, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 220, 40), 2)
    if result is None:
        cv2.putText(
            frame,
            "No body detected",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 255),
            2,
        )
        return

    color = (0, 200, 0) if result.level == "낮음" else (0, 180, 255) if result.level == "중간" else (0, 0, 255)
    cv2.putText(
        frame,
        f"Risk: {result.score}/100 ({result.level})",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )

    y = 100
    for line in result.findings[:3]:
        cv2.putText(frame, f"- {line}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 24


def print_routine(result: AnalysisResult, mode: str) -> None:
    print("\n" + "=" * 70)
    print(f"[{mode}] 위험도 {result.score}/100 ({result.level})")
    print("주요 소견:")
    for f in result.findings:
        print(f" - {f}")
    print("추천 예방 운동:")
    for ex in result.exercises:
        print(f" - {ex}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    analyzer = MusculoskeletalAnalyzer(gait_window=args.window)

    # On some Windows setups, MSMF opens the camera but fails to read frames.
    # Prefer DirectShow first, then fall back to default backend.
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. --camera 인덱스를 확인하세요.")

    print(
        "실행 중: 종료하려면 q 키를 누르세요.\n"
        "주의: 본 프로그램은 의학적 진단이 아닌 예방 참고용입니다."
    )

    last_print_ts = 0.0

    with MP_POSE.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            analysis: Optional[AnalysisResult] = None
            lmset = landmarks_from_result(result.pose_landmarks)
            if lmset is not None:
                if args.mode == "sitting":
                    analysis = analyzer.analyze_sitting(lmset)
                else:
                    analysis = analyzer.analyze_gait_frame(lmset)

                MP_DRAW.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    MP_POSE.POSE_CONNECTIONS,
                    landmark_drawing_spec=MP_DRAW.DrawingSpec(color=(100, 255, 100), thickness=2, circle_radius=2),
                    connection_drawing_spec=MP_DRAW.DrawingSpec(color=(255, 180, 60), thickness=2, circle_radius=2),
                )

            draw_overlay(frame, args.mode, analysis)
            cv2.imshow("Musculoskeletal Assistant", frame)

            now = time.time()
            if analysis is not None and now - last_print_ts > 5.0:
                print_routine(analysis, args.mode)
                last_print_ts = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
