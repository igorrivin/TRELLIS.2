import base64
import mimetypes
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests


THREEDAI_API_BASE_URL = os.environ.get("THREEDAI_API_BASE_URL", "https://api.3daistudio.com").rstrip("/")
THREEDAI_REQUEST_TIMEOUT = float(os.environ.get("THREEDAI_REQUEST_TIMEOUT", "120"))
THREEDAI_DOWNLOAD_TIMEOUT = float(os.environ.get("THREEDAI_DOWNLOAD_TIMEOUT", "300"))
THREEDAI_POLL_INTERVAL = float(os.environ.get("THREEDAI_POLL_INTERVAL", "15"))
THREEDAI_GENERATION_TIMEOUT = float(os.environ.get("THREEDAI_GENERATION_TIMEOUT", "3600"))


class ThreeDAIStudioError(RuntimeError):
    pass


@dataclass
class ThreeDAIStudioResult:
    task_id: str
    glb_path: str
    asset_url: str


def get_threedai_api_key() -> Optional[str]:
    return (
        os.environ.get("THREEDAI_API_KEY")
        or os.environ.get("THREEDAISTUDIO_API_KEY")
        or os.environ.get("AI3DSTUDIO_API_KEY")
        or os.environ.get("THREEDAI_STUDIO_API_KEY")
        or os.environ.get("3DAISTUDIO_API_KEY")
    )


def generate_tripo_p1(
    image_paths: list[str],
    output_dir: str,
    api_key: str,
    *,
    prompt: str = "",
    negative_prompt: str = "",
    face_limit: int = 10000,
    texture: bool = True,
    pbr: bool = True,
    texture_quality: str = "standard",
    model_seed: Optional[int] = None,
    image_seed: Optional[int] = None,
    texture_seed: Optional[int] = None,
    auto_size: bool = False,
    export_uv: bool = True,
    compress_geometry: bool = False,
    texture_alignment: str = "original_image",
    orientation: str = "align_image",
    enable_image_autofix: bool = False,
    poll_interval: float = THREEDAI_POLL_INTERVAL,
    timeout: float = THREEDAI_GENERATION_TIMEOUT,
    progress: Optional[Callable] = None,
) -> ThreeDAIStudioResult:
    os.makedirs(output_dir, exist_ok=True)
    if not image_paths and not prompt.strip():
        raise ThreeDAIStudioError("Tripo P1 text-to-3D requires a prompt when no images are provided.")
    if len(image_paths) > 4:
        raise ThreeDAIStudioError("Tripo P1 supports at most 4 multiview images.")

    endpoint, payload = build_tripo_p1_payload(
        image_paths,
        prompt=prompt,
        negative_prompt=negative_prompt,
        face_limit=face_limit,
        texture=texture,
        pbr=pbr,
        texture_quality=texture_quality,
        model_seed=model_seed,
        image_seed=image_seed,
        texture_seed=texture_seed,
        auto_size=auto_size,
        export_uv=export_uv,
        compress_geometry=compress_geometry,
        texture_alignment=texture_alignment,
        orientation=orientation,
        enable_image_autofix=enable_image_autofix,
    )

    update_progress(progress, 0.02, "Submitting to Tripo P1")
    task_id = submit_generation(api_key, endpoint, payload)
    wait_until_done(api_key, task_id, poll_interval=poll_interval, timeout=timeout, progress=progress)
    update_progress(progress, 0.95, "Downloading Tripo P1 result")
    return download_result(api_key, task_id, output_dir)


def build_tripo_p1_payload(
    image_paths: list[str],
    *,
    prompt: str,
    negative_prompt: str,
    face_limit: int,
    texture: bool,
    pbr: bool,
    texture_quality: str,
    model_seed: Optional[int],
    image_seed: Optional[int],
    texture_seed: Optional[int],
    auto_size: bool,
    export_uv: bool,
    compress_geometry: bool,
    texture_alignment: str,
    orientation: str,
    enable_image_autofix: bool,
) -> tuple[str, dict]:
    shared = {
        "face_limit": int(face_limit),
        "texture": bool(texture),
        "pbr": bool(pbr),
        "texture_quality": texture_quality,
        "auto_size": bool(auto_size),
        "export_uv": bool(export_uv),
    }
    if model_seed is not None:
        shared["model_seed"] = int(model_seed)
    if texture_seed is not None:
        shared["texture_seed"] = int(texture_seed)
    if compress_geometry:
        shared["compress"] = "geometry"

    if not image_paths:
        payload = {**shared, "prompt": prompt.strip()}
        if negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt.strip()
        if image_seed is not None:
            payload["image_seed"] = int(image_seed)
        return "/v1/3d-models/tripo/text-to-3d/p1/", payload

    image_options = {
        **shared,
        "texture_alignment": texture_alignment,
        "orientation": orientation,
    }
    if len(image_paths) == 1:
        payload = {
            **image_options,
            "image": image_to_data_uri(image_paths[0]),
            "enable_image_autofix": bool(enable_image_autofix),
        }
        return "/v1/3d-models/tripo/image-to-3d/p1/", payload

    payload = {
        **image_options,
        "images": [{"image": image_to_data_uri(path)} for path in image_paths],
    }
    return "/v1/3d-models/tripo/multiview-to-3d/p1/", payload


def image_to_data_uri(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def submit_generation(api_key: str, endpoint: str, payload: dict) -> str:
    data = request_json(
        "POST",
        endpoint,
        api_key=api_key,
        json=payload,
        timeout=THREEDAI_REQUEST_TIMEOUT,
    )
    task_id = data.get("task_id")
    if not task_id:
        raise ThreeDAIStudioError(f"Tripo P1 submit response did not include task_id: {data}")
    return task_id


def wait_until_done(
    api_key: str,
    task_id: str,
    *,
    poll_interval: float,
    timeout: float,
    progress: Optional[Callable],
) -> None:
    started = time.monotonic()
    last_status_text = ""
    while True:
        data = query_task(api_key, task_id)
        status = str(data.get("status", "UNKNOWN")).upper()
        api_progress = data.get("progress")
        elapsed = time.monotonic() - started
        fraction = min(0.9, 0.08 + 0.8 * (elapsed / timeout))
        if isinstance(api_progress, (int, float)):
            fraction = min(0.9, 0.08 + 0.8 * (float(api_progress) / 100.0))

        status_text = f"{status} {api_progress}%" if api_progress is not None else status
        if status_text != last_status_text:
            update_progress(progress, fraction, f"Tripo P1 status: {status_text}")
            last_status_text = status_text

        if status == "FINISHED":
            update_progress(progress, 0.92, "Tripo P1 generation complete")
            return
        if status in {"FAILED", "FAILURE", "ERROR", "CANCELED", "CANCELLED"}:
            failure = data.get("failure_reason") or data
            raise ThreeDAIStudioError(f"Tripo P1 generation failed: {failure}")
        if elapsed > timeout:
            raise ThreeDAIStudioError(f"Timed out waiting for Tripo P1 after {int(timeout)} seconds.")

        time.sleep(max(1.0, poll_interval))


def query_task(api_key: str, task_id: str) -> dict:
    return request_json(
        "GET",
        f"/v1/generation-request/{task_id}/status/",
        api_key=api_key,
        timeout=THREEDAI_REQUEST_TIMEOUT,
    )


def download_result(api_key: str, task_id: str, output_dir: str) -> ThreeDAIStudioResult:
    data = query_task(api_key, task_id)
    results = data.get("results") or []
    asset_url = None
    for item in results:
        if item.get("asset") and item.get("asset_type") == "3D_MODEL":
            asset_url = item["asset"]
            break
    if asset_url is None:
        for item in results:
            if item.get("asset"):
                asset_url = item["asset"]
                break
    if not asset_url:
        raise ThreeDAIStudioError(f"Tripo P1 task completed without a model URL: {data}")

    glb_path = os.path.join(output_dir, "tripo_p1_result.glb")
    download_file(asset_url, glb_path)
    return ThreeDAIStudioResult(task_id=task_id, glb_path=glb_path, asset_url=asset_url)


def request_json(method: str, endpoint: str, *, api_key: str, **kwargs) -> dict:
    headers = kwargs.pop("headers", {})
    headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    response = requests.request(method, f"{THREEDAI_API_BASE_URL}{endpoint}", headers=headers, **kwargs)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text[:1000]
        raise ThreeDAIStudioError(f"3D AI Studio request failed with HTTP {response.status_code}: {detail}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise ThreeDAIStudioError(f"3D AI Studio returned non-JSON response: {response.text[:1000]}") from exc


def download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=THREEDAI_DOWNLOAD_TIMEOUT) as response:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:1000]
            raise ThreeDAIStudioError(f"Tripo P1 file download failed with HTTP {response.status_code}: {detail}") from exc
        with open(path, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out_file.write(chunk)


def update_progress(progress: Optional[Callable], value: float, desc: str) -> None:
    if progress is None:
        return
    try:
        progress(value, desc=desc)
    except TypeError:
        progress(value)
