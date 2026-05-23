import base64
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests


HITEM3D_API_BASE_URL = os.environ.get("HITEM3D_API_BASE_URL", "https://api.hitem3d.ai").rstrip("/")
HITEM3D_REQUEST_TIMEOUT = float(os.environ.get("HITEM3D_REQUEST_TIMEOUT", "120"))
HITEM3D_DOWNLOAD_TIMEOUT = float(os.environ.get("HITEM3D_DOWNLOAD_TIMEOUT", "300"))
HITEM3D_POLL_INTERVAL = float(os.environ.get("HITEM3D_POLL_INTERVAL", "15"))
HITEM3D_GENERATION_TIMEOUT = float(os.environ.get("HITEM3D_GENERATION_TIMEOUT", "3600"))


class Hitem3DError(RuntimeError):
    pass


@dataclass
class Hitem3DResult:
    task_id: str
    glb_path: str
    cover_path: Optional[str]


def has_hitem3d_credentials() -> bool:
    return bool(os.environ.get("HITEM3D_CLIENT_ID") and os.environ.get("HITEM3D_CLIENT_SECRET"))


def get_access_token() -> str:
    client_id = os.environ.get("HITEM3D_CLIENT_ID")
    client_secret = os.environ.get("HITEM3D_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise Hitem3DError("Set HITEM3D_CLIENT_ID and HITEM3D_CLIENT_SECRET before using Hitem3D.")

    raw = f"{client_id}:{client_secret}".encode("utf-8")
    basic = base64.b64encode(raw).decode("ascii")
    payload = request_json(
        "POST",
        "/open-api/v1/auth/token",
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        },
        timeout=HITEM3D_REQUEST_TIMEOUT,
    )
    token = (payload.get("data") or {}).get("accessToken")
    if not token:
        raise Hitem3DError(f"Hitem3D token response did not include accessToken: {payload}")
    return token


def generate_images_to_3d(
    image_paths: list[str],
    output_dir: str,
    *,
    model: str,
    resolution: str,
    face_count: int,
    pbr: bool = True,
    progress: Optional[Callable] = None,
) -> Hitem3DResult:
    if not image_paths:
        raise Hitem3DError("Hitem3D requires at least one image.")
    if len(image_paths) > 4:
        raise Hitem3DError("Hitem3D supports at most 4 multi-view images.")

    os.makedirs(output_dir, exist_ok=True)
    token = get_access_token()
    update_progress(progress, 0.02, "Submitting to Hitem3D")
    task_id = submit_task(
        token,
        image_paths,
        model=model,
        resolution=resolution,
        face_count=face_count,
        pbr=pbr,
    )

    wait_until_done(token, task_id, progress=progress)
    update_progress(progress, 0.95, "Downloading Hitem3D result")
    return download_result(token, task_id, output_dir)


def submit_task(
    token: str,
    image_paths: list[str],
    *,
    model: str,
    resolution: str,
    face_count: int,
    pbr: bool,
) -> str:
    handles = []
    try:
        files = []
        file_field = "images" if len(image_paths) == 1 else "multi_images"
        for path in image_paths:
            handle = open(path, "rb")
            handles.append(handle)
            files.append((file_field, (os.path.basename(path), handle, "image/png")))

        data = {
            "request_type": "3",
            "model": model,
            "resolution": resolution,
            "face": str(int(face_count)),
            "format": "2",
            "pbr": "1" if pbr else "0",
        }
        if len(image_paths) > 1:
            data["multi_images_bit"] = ("1" * len(image_paths)).ljust(4, "0")

        payload = request_json(
            "POST",
            "/open-api/v1/submit-task",
            headers={"Authorization": f"Bearer {token}", "Accept": "*/*"},
            files=files,
            data=data,
            timeout=HITEM3D_REQUEST_TIMEOUT,
        )
    finally:
        for handle in handles:
            handle.close()

    task_id = (payload.get("data") or {}).get("task_id")
    if not task_id:
        raise Hitem3DError(f"Hitem3D submit response did not include task_id: {payload}")
    return task_id


def wait_until_done(token: str, task_id: str, *, progress: Optional[Callable]) -> None:
    started = time.monotonic()
    last_state = ""
    while True:
        data = query_task(token, task_id)
        state = str(data.get("state", "unknown"))
        if state != last_state:
            update_progress(progress, min(0.9, 0.05 + 0.8 * ((time.monotonic() - started) / HITEM3D_GENERATION_TIMEOUT)), f"Hitem3D status: {state}")
            last_state = state

        if state == "success":
            update_progress(progress, 0.92, "Hitem3D generation complete")
            return
        if state == "failed":
            raise Hitem3DError(f"Hitem3D task failed: {data}")
        if time.monotonic() - started > HITEM3D_GENERATION_TIMEOUT:
            raise Hitem3DError(f"Timed out waiting for Hitem3D after {int(HITEM3D_GENERATION_TIMEOUT)} seconds.")

        time.sleep(max(5.0, HITEM3D_POLL_INTERVAL))


def query_task(token: str, task_id: str) -> dict:
    payload = request_json(
        "GET",
        "/open-api/v1/query-task",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        },
        params={"task_id": task_id},
        timeout=HITEM3D_REQUEST_TIMEOUT,
    )
    return payload.get("data") or {}


def download_result(token: str, task_id: str, output_dir: str) -> Hitem3DResult:
    data = query_task(token, task_id)
    model_url = data.get("url")
    cover_url = data.get("cover_url")
    if not model_url:
        raise Hitem3DError(f"Hitem3D task completed without a model URL: {data}")

    glb_path = os.path.join(output_dir, "hitem3d_result.glb")
    download_file(model_url, glb_path)

    cover_path = None
    if cover_url:
        cover_path = os.path.join(output_dir, "hitem3d_cover.webp")
        download_file(cover_url, cover_path)

    return Hitem3DResult(task_id=task_id, glb_path=glb_path, cover_path=cover_path)


def request_json(method: str, endpoint: str, **kwargs) -> dict:
    response = requests.request(method, f"{HITEM3D_API_BASE_URL}{endpoint}", **kwargs)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text[:1000]
        raise Hitem3DError(f"Hitem3D request failed with HTTP {response.status_code}: {detail}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise Hitem3DError(f"Hitem3D returned non-JSON response: {response.text[:1000]}") from exc

    if payload.get("code") != 200:
        raise Hitem3DError(f"Hitem3D API error {payload.get('code')}: {payload.get('msg') or payload}")
    return payload


def download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=HITEM3D_DOWNLOAD_TIMEOUT) as response:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:1000]
            raise Hitem3DError(f"Hitem3D file download failed with HTTP {response.status_code}: {detail}") from exc
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
