from __future__ import annotations

import csv
import io
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import streamlit as st

from analyzer import AnalysisResult, MusculoskeletalAnalyzer
from pose_utils import LandmarkSet, to_point

# Mediapipe imports matplotlib internally; set cache dir to writable workspace path.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

import mediapipe as mp


MP_POSE = mp.solutions.pose
MP_DRAW = mp.solutions.drawing_utils


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


def level_color(level: str) -> str:
    if level in {"낮음", "??쓬"}:
        return "green"
    if level in {"중간", "以묎컙"}:
        return "orange"
    return "red"


def decode_uploaded_image(file) -> Optional[np.ndarray]:
    data = np.frombuffer(file.read(), dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def run_app() -> None:
    st.set_page_config(page_title="근골격계 자세 분석 (프로토타입)", layout="wide")
    st.title("근골격계 자세 분석 프로토타입")
    st.caption("카메라 없이 사진 1장 업로드로 동작합니다. 의료 진단이 아닌 참고용입니다.")

    if "saved_results" not in st.session_state:
        st.session_state.saved_results = []

    col1, col2 = st.columns([1.5, 1.5])
    with col1:
        mode = st.selectbox("분석 모드", ["sitting", "gait"], index=0)
    with col2:
        gait_window = st.slider("보행 분석 윈도우", min_value=30, max_value=240, value=120, step=10)

    uploaded = st.file_uploader(
        "분석할 사진을 업로드하세요 (JPG/PNG/WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    save_col, clear_col, dl_col = st.columns([1, 1, 1.4])

    if uploaded is None:
        st.info("사진을 업로드하면 즉시 분석 결과를 보여줍니다.")
    else:
        frame = decode_uploaded_image(uploaded)
        if frame is None:
            st.error("이미지를 읽지 못했습니다. 다른 파일로 다시 시도해 주세요.")
        else:
            analyzer = MusculoskeletalAnalyzer(gait_window=gait_window)
            preview = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with MP_POSE.Pose(
                static_image_mode=True,
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            ) as pose:
                result = pose.process(rgb)

            lmset = landmarks_from_result(result.pose_landmarks)
            if lmset is None:
                st.warning("신체 랜드마크를 찾지 못했습니다. 전신이 잘 보이는 사진으로 다시 시도해 주세요.")
                st.image(rgb, caption="업로드 원본", use_container_width=True)
            else:
                analysis: AnalysisResult
                if mode == "sitting":
                    analysis = analyzer.analyze_sitting(lmset)
                else:
                    analysis = analyzer.analyze_gait_frame(lmset)

                MP_DRAW.draw_landmarks(
                    preview,
                    result.pose_landmarks,
                    MP_POSE.POSE_CONNECTIONS,
                    landmark_drawing_spec=MP_DRAW.DrawingSpec(color=(100, 255, 100), thickness=2, circle_radius=2),
                    connection_drawing_spec=MP_DRAW.DrawingSpec(color=(255, 180, 60), thickness=2, circle_radius=2),
                )

                v1, v2 = st.columns(2)
                with v1:
                    st.image(rgb, caption="업로드 원본", use_container_width=True)
                with v2:
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="랜드마크 오버레이", use_container_width=True)

                color = level_color(analysis.level)
                st.markdown(f"### 위험도: :{color}[{analysis.score}/100 ({analysis.level})]")
                st.write({"주요 소견": analysis.findings})
                st.write({"추천 운동": analysis.exercises})
                st.expander("세부 지표").write(analysis.metrics)

                if mode == "gait":
                    st.caption("참고: 사진 1장 기반 보행 분석은 정확도가 낮습니다. 프로토타입 확인용으로만 사용하세요.")

                if save_col.button("현재 결과 저장", use_container_width=True):
                    st.session_state.saved_results.append(
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "mode": mode,
                            "score": analysis.score,
                            "level": analysis.level,
                            "findings": " | ".join(analysis.findings),
                            "exercises": " | ".join(analysis.exercises),
                            "metrics": str(analysis.metrics),
                            "file": uploaded.name,
                        }
                    )
                    st.success("현재 결과를 저장했습니다.")

    if clear_col.button("기록 초기화", use_container_width=True):
        st.session_state.saved_results = []
        st.info("저장 기록을 초기화했습니다.")

    if st.session_state.saved_results:
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=[
                "timestamp",
                "mode",
                "score",
                "level",
                "findings",
                "exercises",
                "metrics",
                "file",
            ],
        )
        writer.writeheader()
        writer.writerows(st.session_state.saved_results)
        dl_col.download_button(
            label="CSV 다운로드",
            data=csv_buffer.getvalue().encode("utf-8-sig"),
            file_name="musculoskeletal_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state.saved_results:
        st.caption(f"저장된 기록: {len(st.session_state.saved_results)}건")


if __name__ == "__main__":
    run_app()
