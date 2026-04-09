from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

from pose_utils import (
    LandmarkSet,
    angle_abc,
    midpoint,
    rolling_mean,
    rolling_std,
    vertical_deviation,
)


@dataclass
class AnalysisResult:
    score: int
    level: str
    findings: List[str]
    exercises: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)


class MusculoskeletalAnalyzer:
    """
    Rule-based estimator for posture and gait risk.
    Not a medical diagnosis.
    """

    def __init__(self, gait_window: int = 120) -> None:
        self.gait_window = gait_window
        self.left_knee_angles: Deque[float] = deque(maxlen=gait_window)
        self.right_knee_angles: Deque[float] = deque(maxlen=gait_window)
        self.pelvis_shift: Deque[float] = deque(maxlen=gait_window)
        self.trunk_tilt: Deque[float] = deque(maxlen=gait_window)

    @staticmethod
    def _level(score: int) -> str:
        if score < 35:
            return "낮음"
        if score < 65:
            return "중간"
        return "높음"

    def analyze_sitting(self, lm: LandmarkSet) -> AnalysisResult:
        l_sh = lm.get("left_shoulder")
        r_sh = lm.get("right_shoulder")
        l_hip = lm.get("left_hip")
        r_hip = lm.get("right_hip")
        l_ear = lm.get("left_ear")
        r_ear = lm.get("right_ear")
        l_knee = lm.get("left_knee")
        r_knee = lm.get("right_knee")

        shoulder_mid = midpoint(l_sh, r_sh)
        hip_mid = midpoint(l_hip, r_hip)
        ear_mid = midpoint(l_ear, r_ear)
        knee_mid = midpoint(l_knee, r_knee)

        neck_forward = abs(vertical_deviation(ear_mid, shoulder_mid))
        trunk_angle = angle_abc(shoulder_mid, hip_mid, knee_mid)
        pelvic_obliquity = abs(l_hip[1] - r_hip[1])

        score = 15
        findings: List[str] = []

        if neck_forward > 0.03:
            score += 30
            findings.append("목 전방 자세(거북목) 위험이 높습니다.")
        elif neck_forward > 0.02:
            score += 18
            findings.append("목 전방 자세 경향이 보입니다.")

        # Hip-centered angle near 180 is neutral; away from it means flexion/extension bias.
        trunk_bias = abs(180.0 - trunk_angle)
        if trunk_bias > 32:
            score += 24
            findings.append("허리-골반 정렬이 크게 무너져 있습니다.")
        elif trunk_bias > 20:
            score += 14
            findings.append("허리-골반 정렬 불균형이 관찰됩니다.")

        if pelvic_obliquity > 0.04:
            score += 18
            findings.append("골반 좌우 높이 차가 큽니다.")
        elif pelvic_obliquity > 0.025:
            score += 10
            findings.append("가벼운 골반 비대칭이 보입니다.")

        score = max(0, min(100, score))
        level = self._level(score)
        exercises = self._exercise_plan(
            neck_risk=neck_forward > 0.02,
            trunk_risk=trunk_bias > 20,
            pelvic_risk=pelvic_obliquity > 0.025,
            gait_risk=False,
        )
        if not findings:
            findings = ["큰 이상은 보이지 않지만 장시간 고정 자세는 피하세요."]

        return AnalysisResult(
            score=score,
            level=level,
            findings=findings,
            exercises=exercises,
            metrics={
                "neck_forward": neck_forward,
                "trunk_bias": trunk_bias,
                "pelvic_obliquity": pelvic_obliquity,
            },
        )

    def analyze_gait_frame(self, lm: LandmarkSet) -> AnalysisResult:
        l_sh = lm.get("left_shoulder")
        r_sh = lm.get("right_shoulder")
        l_hip = lm.get("left_hip")
        r_hip = lm.get("right_hip")
        l_knee = lm.get("left_knee")
        r_knee = lm.get("right_knee")
        l_ankle = lm.get("left_ankle")
        r_ankle = lm.get("right_ankle")

        left_knee_angle = angle_abc(l_hip, l_knee, l_ankle)
        right_knee_angle = angle_abc(r_hip, r_knee, r_ankle)
        pelvis_y_diff = l_hip[1] - r_hip[1]
        shoulder_mid = midpoint(l_sh, r_sh)
        hip_mid = midpoint(l_hip, r_hip)
        trunk_center_tilt = vertical_deviation(shoulder_mid, hip_mid)

        self.left_knee_angles.append(left_knee_angle)
        self.right_knee_angles.append(right_knee_angle)
        self.pelvis_shift.append(pelvis_y_diff)
        self.trunk_tilt.append(trunk_center_tilt)

        knee_asym = abs(rolling_mean(self.left_knee_angles) - rolling_mean(self.right_knee_angles))
        pelvis_sway = rolling_std(self.pelvis_shift)
        trunk_instability = rolling_std(self.trunk_tilt)

        score = 20
        findings: List[str] = []

        if knee_asym > 14:
            score += 26
            findings.append("좌우 무릎 움직임 비대칭이 큽니다.")
        elif knee_asym > 9:
            score += 14
            findings.append("좌우 무릎 움직임 비대칭 경향이 있습니다.")

        if pelvis_sway > 0.020:
            score += 24
            findings.append("골반 좌우 흔들림이 커 고관절 안정성이 낮을 수 있습니다.")
        elif pelvis_sway > 0.013:
            score += 12
            findings.append("골반 흔들림이 다소 큽니다.")

        if trunk_instability > 0.018:
            score += 20
            findings.append("보행 시 몸통 안정성이 낮습니다.")
        elif trunk_instability > 0.011:
            score += 10
            findings.append("몸통 기울기 변동이 관찰됩니다.")

        score = max(0, min(100, score))
        level = self._level(score)
        gait_risk = knee_asym > 9 or pelvis_sway > 0.013 or trunk_instability > 0.011
        exercises = self._exercise_plan(
            neck_risk=False,
            trunk_risk=trunk_instability > 0.011,
            pelvic_risk=pelvis_sway > 0.013,
            gait_risk=gait_risk,
        )

        if not findings:
            findings = ["현재 보행 패턴에서 큰 이상은 보이지 않습니다."]

        return AnalysisResult(
            score=score,
            level=level,
            findings=findings,
            exercises=exercises,
            metrics={
                "knee_asymmetry": knee_asym,
                "pelvis_sway_std": pelvis_sway,
                "trunk_instability_std": trunk_instability,
            },
        )

    def _exercise_plan(
        self,
        *,
        neck_risk: bool,
        trunk_risk: bool,
        pelvic_risk: bool,
        gait_risk: bool,
    ) -> List[str]:
        plan: List[str] = []
        if neck_risk:
            plan.append("턱 당기기(Chin tuck) 10회 x 3세트")
            plan.append("흉추 신전 스트레칭 30초 x 3회")
        if trunk_risk:
            plan.append("버드독(Bird-dog) 좌우 10회 x 3세트")
            plan.append("플랭크 20~40초 x 3세트")
        if pelvic_risk:
            plan.append("클램셸(중둔근 강화) 12회 x 3세트")
            plan.append("힙 브릿지 12회 x 3세트")
        if gait_risk:
            plan.append("싱글 레그 스탠스 30초 x 3세트")
            plan.append("사이드 스텝 밴드 워크 10m x 3회")
        if not plan:
            plan.append("가벼운 전신 스트레칭 10분/일")
            plan.append("1시간마다 2~3분 자세 리셋")
        return plan

