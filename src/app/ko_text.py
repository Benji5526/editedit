"""Korean names for the program's own values (mosaic styles, job states, review event types and actions). The values themselves
stay as they are in files, logs and the command line; only what the user reads is translated here."""
from __future__ import annotations

import re

STYLE_TEXT = {"pixelate": "픽셀 모자이크", "blur": "흐림", "solid": "단색 덮기"}

STATUS_TEXT = {"completed_no_flags": "완료 (검수할 항목 없음)", "completed_with_review_required": "완료 (검수 필요)",
               "cancelled": "취소됨", "failed": "실패", "planned": "계획만 세움", "interrupted": "중단됨", "running": "처리 중"}

# review events: short name shown in the table, and one sentence that says what it means
EVENT_TEXT = {
    "tracking_lost_mid_frame": ("추적 끊김 (화면 안쪽)", "따라가던 대상의 영역이 화면 안쪽에서 갑자기 사라졌습니다. 추적을 놓쳤을 수 있으니 확인하세요."),
    "tracking_lost_at_edge": ("추적 끊김 (화면 가장자리)", "대상의 영역이 화면 가장자리에서 사라졌습니다. 화면 밖으로 나간 것일 수 있습니다."),
    "tracking_recovered": ("추적 다시 잡힘", "끊겼던 대상의 영역이 다시 잡혔습니다. 끊겼던 동안이 제대로 가려졌는지 확인하세요."),
    "mask_area_jump": ("영역 크기 급변", "가리는 영역의 크기가 앞 프레임과 크게 달라졌습니다. 다른 것을 잡았거나 일부를 놓쳤을 수 있습니다."),
    "detection_without_mask": ("탐지됐지만 추적 영역 밖", "자동 탐지가 찾은 곳이 추적 중인 어떤 대상과도 겹치지 않습니다. 두 곳 모두 가려집니다."),
    "detection_lost": ("탐지 끊김", "자동 탐지로 잡히던 곳이 갑자기 잡히지 않습니다. 화면에서 사라졌거나 탐지를 놓친 것입니다."),
    "chunk_boundary_resumed": ("조각 경계에서 이어 붙임", "조각이 바뀌는 지점에서 추적을 정상적으로 이어 갔습니다."),
    "chunk_boundary_resumed_from_stale_mask": ("조각 경계에서 예전 영역으로 이어 붙임", "조각이 바뀌는 지점에 대상이 보이지 않아, 그보다 앞선 프레임의 영역으로 추적을 다시 시작했습니다."),
    "chunk_boundary_resume_failed": ("조각 경계에서 이어 붙이지 못함", "조각이 바뀌는 지점에서 추적을 이어 가지 못했습니다. 이 조각에서 대상이 가려지지 않았을 수 있습니다."),
    "tracking_abandoned": ("추적 포기", "대상이 오랫동안 보이지 않아 더 이상 따라가지 않습니다. 다시 나타나는 곳에서 클릭을 추가하세요."),
    "model_error": ("추적 모델 오류", "추적 모델이 이 구간을 처리하지 못했습니다."),
    "detector_error": ("탐지 모델 오류", "자동 탐지 모델이 이 프레임을 처리하지 못했습니다."),
    "frame_error": ("프레임 읽기/쓰기 오류", "영상 프레임을 읽거나 쓰지 못했습니다."),
    "frame_count_mismatch": ("프레임 수 불일치", "결과의 프레임 수가 원본과 다릅니다."),
    "output_verification_failed": ("결과 영상 검증 실패", "만들어진 결과 영상이 검증을 통과하지 못했습니다."),
    "fail_closed_mosaic_applied": ("안전 모드로 넓게 가림", "확실하지 않아서 평소보다 넓은 범위를 가렸습니다."),
    "timestamp_warning": ("시간 정보 경고", "영상의 시간 정보가 일정하지 않습니다."),
    "audio_reencoded": ("소리 다시 압축", "원본의 소리 형식을 그대로 넣을 수 없어 AAC로 다시 압축했습니다."),
    "prompt_produced_no_mask": ("클릭한 곳에서 영역을 못 찾음", "클릭한 지점에서 추적 모델이 가릴 영역을 만들지 못했습니다. 다른 곳을 클릭하거나 브러시로 칠하세요."),
}

ACTION_TEXT = {
    "logged": "기록만 함", "flagged": "검수 표시", "flagged_only": "검수 표시만 (가리지 않음)",
    "flagged_no_region_known": "검수 표시 (가릴 위치를 알 수 없음)",
    "expanded_bbox_mosaic": "마지막 위치를 넓혀 가림", "full_frame_mosaic": "화면 전체를 가림", "fail_closed_full_frame": "화면 전체를 가림",
    "mosaicked_from_detection_box": "탐지된 네모 영역으로 가림", "object_no_longer_tracked": "이 대상 추적을 끝냄",
    "object_not_tracked_in_chunk": "이 조각에서는 추적하지 못함", "rerun_from_last_good_frame": "마지막으로 잡혔던 프레임부터 다시 추적",
    "resumed": "이어 감", "resumed_normal_unverified": "이어 감 (확인되지 않음)", "job_aborted": "작업 중단",
    "output_not_finalised": "결과 영상을 완성하지 않음", "audio_reencoded_to_aac": "소리를 AAC로 다시 압축",
}


def style_text(value) -> str:
    return STYLE_TEXT.get(value, str(value))


def status_text(value) -> str:
    return STATUS_TEXT.get(value, str(value or "").replace("_", " "))


def event_text(value) -> str:
    return EVENT_TEXT.get(value, (str(value), ""))[0]


def event_help(value) -> str:
    return EVENT_TEXT.get(value, ("", ""))[1]


def action_text(value) -> str:
    m = re.fullmatch(r"expanded_bbox_mosaic_up_to_(\d+)_frames", str(value))
    if m:
        return f"마지막 위치를 넓혀 가림 (최대 {m.group(1)}프레임)"
    return ACTION_TEXT.get(value, str(value))
