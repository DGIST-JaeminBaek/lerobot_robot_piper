"""정책 추론의 공통 runtime: observation 변환, 정책 load, action chunk 예측."""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
from contextlib import nullcontext

import torch
import numpy as np
from dataclasses import dataclass

from hamlet_history import HamletHistorySnapshot


@dataclass(frozen=True)
class CameraCrop:
    x: int
    y: int
    size: int


def parse_camera_crop(value: str | CameraCrop) -> CameraCrop:
    if isinstance(value, CameraCrop): return value
    try: x, y, size = (int(part.strip()) for part in value.split(","))
    except ValueError as error: raise ValueError("Crop must use integer X,Y,SIZE values") from error
    if x < 0 or y < 0 or size <= 0: raise ValueError("Crop x/y must be non-negative and size must be positive")
    return CameraCrop(x, y, size)


def state_from_raw_observation(raw: dict, features: dict) -> np.ndarray:
    names = [name for name in features["observation.state"]["names"] if name.endswith(".pos")]
    if len(names) != 7: raise ValueError(f"This executor requires 7 position fields, got {names}")
    missing = [name for name in names if name not in raw]
    if missing: raise KeyError(f"Raw observation is missing state fields: {missing}")
    return np.asarray([raw[name] for name in names], dtype=np.float32)


def preprocess_live_camera_observation(raw: dict, camera_keys: list[str], crops: dict[str, CameraCrop], output_size: int) -> dict:
    import cv2
    processed = dict(raw)
    for feature_key in camera_keys:
        camera = feature_key.removeprefix("observation.images.")
        if camera not in crops or camera not in raw: raise KeyError(f"Missing live crop or camera: {camera}")
        image, crop = np.asarray(raw[camera]), crops[camera]
        if image.ndim != 3 or image.shape[2] != 3: raise ValueError(f"Live camera {camera!r} must be HWC RGB")
        if crop.x + crop.size > image.shape[1] or crop.y + crop.size > image.shape[0]: raise ValueError(f"Live camera {camera!r} crop exceeds frame")
        processed[camera] = np.ascontiguousarray(cv2.resize(image[crop.y:crop.y+crop.size, crop.x:crop.x+crop.size], (output_size, output_size), interpolation=cv2.INTER_AREA if crop.size >= output_size else cv2.INTER_LINEAR))
    return processed


def validate_live_camera_output_size(features: dict, camera_keys: list[str], output_size: int) -> None:
    for key in camera_keys:
        if tuple(features[key]["shape"]) not in {(output_size, output_size, 3), (3, output_size, output_size)}:
            raise ValueError(f"{key} training shape is incompatible with {output_size}px preprocessing")


def build_robot_from_env(max_relative_target: float | None):
    """실물 추론/승인 실행이 공통으로 쓰는 Piper follower 설정."""
    from lerobot_robot_piper.config_piper import PiperFollowerConfig
    from lerobot_robot_piper.piper_follower import PiperFollower
    env = os.environ
    boolean = lambda key, default: env.get(key, default).strip().lower() in {"1", "true", "yes", "on"}
    return PiperFollower(PiperFollowerConfig(
        port=env.get("FOLLOWER_PORT", "can_follower"), disable_torque_on_disconnect=boolean("DISABLE_TORQUE_ON_DISCONNECT", "true"),
        park_on_connect=boolean("PARK_ON_CONNECT", "false"), camera_type=env.get("CAMERA_TYPE", "intelrealsense"),
        top_cam_type=env.get("TOP_CAM_TYPE", ""), wrist_cam_type=env.get("WRIST_CAM_TYPE", ""), top_cam=env.get("TOP_CAM", ""), wrist_cam=env.get("WRIST_CAM", ""),
        cam_width=int(env.get("CAM_WIDTH", "1280")), cam_height=int(env.get("CAM_HEIGHT", "720")), camera_fps=int(env.get("FPS", "30")), realsense_use_depth=False,
        realsense_warmup_s=float(env.get("REALSENSE_WARMUP_S", "3.0")), camera_connect_warmup=boolean("CAMERA_CONNECT_WARMUP", "false"), camera_post_connect_wait_s=float(env.get("CAMERA_POST_CONNECT_WAIT_S", "2.0")),
        use_effort=True, safety_enabled=boolean("SAFETY_ENABLED", "true"), safety_effort_limit=float(env.get("SAFETY_EFFORT_LIMIT", "8.0")), safety_on_overload=env.get("SAFETY_ON_OVERLOAD", "park").strip().lower(),
        park_release_mode=env.get("PARK_RELEASE_MODE", "park_lower"), park_release_ramp_s=float(env.get("PARK_RELEASE_RAMP_S", "2.0")), park_release_settle_s=float(env.get("PARK_RELEASE_SETTLE_S", "0.5")), park_release_wrist_rest_deg=float(env.get("PARK_RELEASE_WRIST_REST_DEG", "24.4")),
        max_relative_target=max_relative_target, use_action_offset=False,
    ))


def make_raw_observation(dataset, item: dict) -> dict:
    raw: dict[str, object] = {}
    for name, value in zip(dataset.features["observation.state"]["names"], item["observation.state"].detach().cpu().numpy()):
        raw[name] = float(value)
    for feature_key in dataset.meta.camera_keys:
        raw[feature_key.removeprefix("observation.images.")] = (
            item[feature_key].detach().cpu().clamp(0, 1).permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
        )
    return raw


def load_policy(policy_path: str, dataset_meta, device: str, rename_map: dict | None = None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    config = PreTrainedConfig.from_pretrained(policy_path)
    config.pretrained_path, config.device = policy_path, device
    if rename_map is None:
        path = pathlib.Path(policy_path) / "policy_preprocessor.json"
        if path.is_file():
            with contextlib.suppress(json.JSONDecodeError, OSError, KeyError):
                for step in json.loads(path.read_text(encoding="utf-8"))["steps"]:
                    if step.get("registry_name") == "rename_observations_processor":
                        rename_map = step.get("config", {}).get("rename_map") or None
                        break
    policy = make_policy(config, ds_meta=dataset_meta, rename_map=rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=config.pretrained_path, dataset_stats=dataset_meta.stats,
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    return config, policy, preprocessor, postprocessor


def _prepare_policy_observation(*, raw_observation: dict, dataset_features: dict,
                                preprocessor, device: torch.device, task: str) -> dict:
    from lerobot.datasets.utils import OBS_STR, build_dataset_frame
    from lerobot.policies.utils import prepare_observation_for_inference

    frame = build_dataset_frame(dataset_features, raw_observation, prefix=OBS_STR)
    return preprocessor(prepare_observation_for_inference(frame, device, task=task, robot_type="piper_follower"))


def _hamlet_memory_enabled(policy) -> bool:
    """Recognize both current and pre-``hamlet_enabled`` HAMLET checkpoints."""
    config = getattr(policy, "config", None)
    if bool(getattr(config, "hamlet_enabled", False)):
        return True
    # The completed 315 checkpoints predate the explicit flag but retain their
    # registered policy type in config.json.  Do not require re-training merely
    # to opt them into the rollout-history bridge.
    return str(getattr(config, "type", "")).lower() == "smolvla_hamlet"


def _require_identity_visual_normalization(policy) -> None:
    """The live HWC history bridge is only safe for the current IDENTITY path.

    A non-identity visual processor needs an explicit proof that the historical
    batch has exactly the same processor semantics as the current observation.
    Do not silently feed an incompatible scale into HAMLET's ``* 2 - 1`` path.
    """
    mapping = getattr(policy.config, "normalization_mapping", {})
    mode = mapping.get("VISUAL") if isinstance(mapping, dict) else None
    value = getattr(mode, "value", mode)
    if str(value).lower() != "identity":
        raise ValueError(
            "HAMLET real-history rollout currently requires VISUAL normalization=IDENTITY; "
            f"got {value!r}. Add an explicitly verified history preprocessor before enabling it."
        )


def hamlet_real_history_enabled(policy) -> bool:
    """Whether this policy opts into Piper's real image-history bridge."""
    return _hamlet_memory_enabled(policy)


def validate_hamlet_real_history_policy(policy) -> None:
    """Fail before hardware connection for unsupported HAMLET visual pipelines."""
    if _hamlet_memory_enabled(policy):
        if bool(getattr(policy.config, "compile_model", False)):
            raise ValueError(
                "HAMLET real-history rollout is not enabled with compile_model=True; "
                "its dynamic T-frame request API needs a separately validated compiled graph."
            )
        _require_identity_visual_normalization(policy)


def prepare_hamlet_rollout_history(
    *,
    snapshot: HamletHistorySnapshot,
    current_raw_observation: dict,
    current_observation: dict,
    dataset_features: dict,
    policy,
    preprocessor,
    device: torch.device,
    task: str,
) -> dict:
    """Turn immutable HWC history into HAMLET's post-preprocessor tensor window.

    Previous slots follow the exact ``build_dataset_frame →
    prepare_observation_for_inference → preprocessor`` route used by the current
    observation.  The final slot reuses ``current_observation`` itself, so it is
    bit-identical to the policy request rather than a second conversion.
    """
    if not _hamlet_memory_enabled(policy):
        raise ValueError("rollout history was provided for a policy without enabled HAMLET memory")
    _require_identity_visual_normalization(policy)
    expected_keys = list(policy.config.image_features)
    if not expected_keys:
        raise ValueError("HAMLET policy has no visual input features")
    T = int(getattr(policy.config, "memory_window", 0))
    if len(snapshot.frames) != T or snapshot.is_pad.shape != (T,):
        raise ValueError(
            f"HAMLET history must have {T} slots and is_pad shape {(T,)}, got "
            f"{len(snapshot.frames)} / {snapshot.is_pad.shape}"
        )

    camera_for_feature = {key: key.removeprefix("observation.images.") for key in dataset_features
                          if key.startswith("observation.images.")}
    required_cameras = set(camera_for_feature.values())
    for frame in snapshot.frames:
        if set(frame.images) != required_cameras:
            raise ValueError(
                f"HAMLET history cameras {sorted(frame.images)} do not match dataset cameras "
                f"{sorted(required_cameras)}"
            )
    for feature, camera in camera_for_feature.items():
        if camera not in current_raw_observation:
            raise KeyError(f"current observation is missing camera {camera!r} for {feature}")
        if not np.array_equal(snapshot.frames[-1].images[camera], current_raw_observation[camera]):
            raise ValueError(f"HAMLET history current slot differs from current raw camera {camera!r}")

    processed_slots: list[dict] = []
    for frame in snapshot.frames[:-1]:
        historical_raw = dict(current_raw_observation)
        historical_raw.update(frame.images)
        processed_slots.append(
            _prepare_policy_observation(
                raw_observation=historical_raw,
                dataset_features=dataset_features,
                preprocessor=preprocessor,
                device=device,
                task=task,
            )
        )
    processed_slots.append(current_observation)

    images: dict[str, torch.Tensor] = {}
    for key in expected_keys:
        tensors = []
        for slot in processed_slots:
            if key not in slot:
                raise KeyError(
                    f"HAMLET policy visual key {key!r} is absent after preprocessing. "
                    "Check policy_preprocessor rename_map and dataset camera names."
                )
            tensor = slot[key]
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[0] != 1:
                raise ValueError(f"preprocessed {key!r} must be a (1,C,H,W) tensor, got {getattr(tensor, 'shape', None)}")
            if not tensor.dtype.is_floating_point:
                raise ValueError(f"preprocessed {key!r} must be floating point, got {tensor.dtype}")
            tensors.append(tensor)
        images[key] = torch.stack(tensors, dim=1)

    return {
        "images": images,
        "is_pad": torch.from_numpy(snapshot.is_pad.copy()).unsqueeze(0).to(device=device),
    }


def predict_chunk(*, raw_observation: dict, dataset_features: dict, policy, preprocessor, postprocessor,
                  device: torch.device, task: str,
                  rollout_history: HamletHistorySnapshot | None = None,
                  return_control_points: bool = False):
    """action chunk를 예측한다.

    `return_control_points=True`면 `(chunk, control_points)`를 돌려준다. 제어점은
    smolvla_abp 계열 정책만 내놓으며(그 외에는 None), CCR(연속성 제약 재적합)에
    쓰인다. 제어점에도 chunk와 **같은 postprocessor**를 적용해 같은 공간으로
    맞춘다 -- clamped B-스플라인 기저는 단위 분할이라 제어점에 대한 아핀 변환이
    곡선에 그대로 전달되므로, 역정규화된 제어점으로 재적합해도 정규화 공간에서
    한 것과 결과가 같다. postprocessor는 Unnormalizer + Device 두 단계뿐이라
    무상태이므로 두 번 호출해도 안전하다.
    """
    observation = _prepare_policy_observation(
        raw_observation=raw_observation,
        dataset_features=dataset_features,
        preprocessor=preprocessor,
        device=device,
        task=task,
    )
    policy_history = None
    if rollout_history is not None:
        policy_history = prepare_hamlet_rollout_history(
            snapshot=rollout_history,
            current_raw_observation=raw_observation,
            current_observation=observation,
            dataset_features=dataset_features,
            policy=policy,
            preprocessor=preprocessor,
            device=device,
            task=task,
        )
    with torch.inference_mode(), (torch.autocast(device_type=device.type) if device.type == "cuda" and getattr(policy.config, "use_amp", False) else nullcontext()):
        raw_chunk = (
            policy.predict_action_chunk(observation, rollout_history=policy_history)
            if policy_history is not None
            else policy.predict_action_chunk(observation)
        )
        chunk = torch.stack(
            [postprocessor(raw_chunk[:, i, :]) for i in range(raw_chunk.shape[1])], dim=1
        ).squeeze(0).detach().cpu()
        if not return_control_points:
            return chunk

        raw_ctrl = getattr(policy, "last_control_points", None)
        if raw_ctrl is None:
            return chunk, None
        control_points = torch.stack(
            [postprocessor(raw_ctrl[:, i, :]) for i in range(raw_ctrl.shape[1])], dim=1
        ).squeeze(0).detach().cpu()
        return chunk, control_points
