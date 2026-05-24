import os
import re
import time
import mimetypes
from dataclasses import dataclass
from typing import Callable, Optional

import requests
from PIL import Image


RODIN_API_BASE_URL = os.environ.get("RODIN_API_BASE_URL", "https://api.hyper3d.com/api/v2").rstrip("/")
RODIN_REQUEST_TIMEOUT = float(os.environ.get("RODIN_REQUEST_TIMEOUT", "120"))
RODIN_DOWNLOAD_TIMEOUT = float(os.environ.get("RODIN_DOWNLOAD_TIMEOUT", "300"))


class RodinError(RuntimeError):
    pass


@dataclass
class RodinResult:
    task_uuid: str
    subscription_key: str
    glb_path: str
    preview_path: Optional[str]
    downloaded_files: list[str]


def get_rodin_api_key() -> Optional[str]:
    return os.environ.get("HYPER3D_API_KEY") or os.environ.get("RODIN_API_KEY")


def generate_image_to_3d(
    image: Image.Image,
    output_dir: str,
    api_key: str,
    *,
    prompt: str = "",
    seed: Optional[int] = None,
    quality: str = "medium",
    mesh_mode: str = "Quad",
    tapose: bool = False,
    use_original_alpha: bool = True,
    hd_texture: bool = False,
    poll_interval: float = 10.0,
    timeout: float = 1800.0,
    progress: Optional[Callable] = None,
) -> RodinResult:
    os.makedirs(output_dir, exist_ok=True)
    input_path = os.path.join(output_dir, "rodin_input.png")
    image.save(input_path)

    return generate_images_to_3d(
        [input_path],
        output_dir,
        api_key,
        prompt=prompt,
        seed=seed,
        tier="Gen-2",
        quality=quality,
        mesh_mode=mesh_mode,
        tapose=tapose,
        use_original_alpha=use_original_alpha,
        geometry_file_format="glb",
        material="PBR",
        preview_render=True,
        hd_texture=hd_texture,
        poll_interval=poll_interval,
        timeout=timeout,
        progress=progress,
    )


def generate_images_to_3d(
    image_paths: list[str],
    output_dir: str,
    api_key: str,
    *,
    prompt: str = "",
    seed: Optional[int] = None,
    tier: str = "Gen-2",
    quality: str = "medium",
    mesh_mode: str = "Quad",
    tapose: bool = False,
    use_original_alpha: bool = True,
    geometry_file_format: str = "glb",
    material: str = "PBR",
    quality_override: Optional[int] = None,
    addons: Optional[list[str]] = None,
    preview_render: bool = True,
    hd_texture: bool = False,
    texture_delight: bool = False,
    texture_mode: Optional[str] = None,
    is_micro: bool = False,
    is_symmetric: bool = False,
    geometry_instruct_mode: Optional[str] = None,
    bbox_condition: Optional[str] = None,
    poll_interval: float = 10.0,
    timeout: float = 1800.0,
    progress: Optional[Callable] = None,
) -> RodinResult:
    os.makedirs(output_dir, exist_ok=True)
    if not image_paths and not prompt.strip():
        raise RodinError("Rodin text-to-3D requires a prompt when no images are provided.")
    if len(image_paths) > 5:
        raise RodinError("Rodin supports at most 5 images.")

    update_progress(progress, 0.02, "Submitting to Rodin")
    task_uuid, subscription_key = submit_generation(
        image_paths,
        api_key,
        prompt=prompt,
        seed=seed,
        tier=tier,
        quality=quality,
        mesh_mode=mesh_mode,
        tapose=tapose,
        use_original_alpha=use_original_alpha,
        geometry_file_format=geometry_file_format,
        material=material,
        quality_override=quality_override,
        addons=addons or [],
        preview_render=preview_render,
        hd_texture=hd_texture,
        texture_delight=texture_delight,
        texture_mode=texture_mode,
        is_micro=is_micro,
        is_symmetric=is_symmetric,
        geometry_instruct_mode=geometry_instruct_mode,
        bbox_condition=bbox_condition,
    )

    wait_until_done(
        api_key,
        subscription_key,
        poll_interval=poll_interval,
        timeout=timeout,
        progress=progress,
    )

    update_progress(progress, 0.95, "Downloading Rodin result")
    return download_result(api_key, task_uuid, subscription_key, output_dir, geometry_file_format=geometry_file_format)


def submit_generation(
    image_paths: list[str],
    api_key: str,
    *,
    prompt: str,
    seed: Optional[int],
    tier: str,
    quality: str,
    mesh_mode: str,
    tapose: bool,
    use_original_alpha: bool,
    geometry_file_format: str,
    material: str,
    quality_override: Optional[int],
    addons: list[str],
    preview_render: bool,
    hd_texture: bool,
    texture_delight: bool,
    texture_mode: Optional[str],
    is_micro: bool,
    is_symmetric: bool,
    geometry_instruct_mode: Optional[str],
    bbox_condition: Optional[str],
) -> tuple[str, str]:
    tapose_field = "TApose" if tier.startswith("Gen-2.5") else "TAPose"
    form_fields: list[tuple[str, tuple[None, str]]] = [
        ("tier", (None, tier)),
        ("geometry_file_format", (None, geometry_file_format)),
        ("material", (None, material)),
        ("quality", (None, quality)),
        ("mesh_mode", (None, mesh_mode)),
        (tapose_field, (None, str(bool(tapose)).lower())),
        ("use_original_alpha", (None, str(bool(use_original_alpha)).lower())),
        ("preview_render", (None, str(bool(preview_render)).lower())),
        ("hd_texture", (None, str(bool(hd_texture)).lower())),
    ]
    if prompt.strip():
        form_fields.append(("prompt", (None, prompt.strip())))
    if seed is not None:
        form_fields.append(("seed", (None, str(int(seed) % 65536))))
    if quality_override is not None:
        form_fields.append(("quality_override", (None, str(int(quality_override)))))
    for addon in addons:
        if addon:
            form_fields.append(("addons", (None, addon)))
    if texture_delight:
        form_fields.append(("texture_delight", (None, "true")))
    if texture_mode:
        form_fields.append(("texture_mode", (None, texture_mode)))
    if is_micro:
        form_fields.append(("is_micro", (None, "true")))
    if is_symmetric:
        form_fields.append(("is_symmetric", (None, "true")))
    if geometry_instruct_mode:
        form_fields.append(("geometry_instruct_mode", (None, geometry_instruct_mode)))
    if bbox_condition and bbox_condition.strip():
        form_fields.append(("bbox_condition", (None, bbox_condition.strip())))

    headers = {"Authorization": f"Bearer {api_key}"}
    handles = []
    try:
        files = []
        for image_path in image_paths:
            image_file = open(image_path, "rb")
            handles.append(image_file)
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
            files.append(("images", (os.path.basename(image_path), image_file, mime_type)))
        files.extend(form_fields)
        data = post_json("/rodin", headers=headers, files=files, timeout=RODIN_REQUEST_TIMEOUT)
    finally:
        for handle in handles:
            handle.close()

    error = data.get("error")
    if error not in (None, "", "OK"):
        raise RodinError(data.get("message") or f"Rodin generation failed: {error}")

    task_uuid = data.get("uuid")
    subscription_key = (data.get("jobs") or {}).get("subscription_key")
    if not task_uuid or not subscription_key:
        raise RodinError(f"Rodin generation response did not include uuid/subscription_key: {data}")
    return task_uuid, subscription_key


def wait_until_done(
    api_key: str,
    subscription_key: str,
    *,
    poll_interval: float,
    timeout: float,
    progress: Optional[Callable],
) -> None:
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    started = time.monotonic()
    last_status_text = ""
    while True:
        data = post_json(
            "/status",
            headers=headers,
            json={"subscription_key": subscription_key},
            timeout=RODIN_REQUEST_TIMEOUT,
        )
        error = data.get("error")
        if error not in (None, "", "OK"):
            raise RodinError(f"Rodin status check failed: {error}")

        jobs = data.get("jobs") or []
        statuses = [str(job.get("status", "Unknown")).strip() for job in jobs]
        status_text = ", ".join(statuses) or "Unknown"
        elapsed = time.monotonic() - started
        fraction = min(0.9, 0.08 + 0.8 * (elapsed / timeout))
        if status_text != last_status_text:
            update_progress(progress, fraction, f"Rodin status: {status_text}")
            last_status_text = status_text

        if statuses and all(status == "Done" for status in statuses):
            update_progress(progress, 0.92, "Rodin generation complete")
            return
        if any(status == "Failed" for status in statuses):
            raise RodinError(f"Rodin generation failed: {data}")
        if elapsed > timeout:
            raise RodinError(f"Timed out waiting for Rodin after {int(timeout)} seconds.")

        time.sleep(max(1.0, poll_interval))


def download_result(
    api_key: str,
    task_uuid: str,
    subscription_key: str,
    output_dir: str,
    *,
    geometry_file_format: str = "glb",
) -> RodinResult:
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = post_json(
        "/download",
        headers=headers,
        json={"task_uuid": task_uuid},
        timeout=RODIN_REQUEST_TIMEOUT,
    )
    error = data.get("error")
    if error not in (None, "", "OK"):
        raise RodinError(f"Rodin download lookup failed: {error}")

    files = data.get("list") or []
    if not files:
        raise RodinError(f"Rodin download response did not include any files: {data}")

    downloaded_files = []
    preview_path = None
    asset_path = None
    expected_ext = f".{geometry_file_format.lower().lstrip('.')}"
    for index, item in enumerate(files):
        url = item.get("url")
        if not url:
            continue
        name = item.get("name") or f"rodin_result_{index}"
        path = download_file(url, output_dir, name, index)
        downloaded_files.append(path)

        lower_path = path.lower()
        is_preview = lower_path.endswith((".png", ".jpg", ".jpeg", ".webp"))
        if lower_path.endswith(expected_ext) and asset_path is None:
            asset_path = path
        if not is_preview and asset_path is None:
            asset_path = path
        if lower_path.endswith((".png", ".jpg", ".jpeg", ".webp")) and preview_path is None:
            preview_path = path

    if asset_path is None:
        raise RodinError(f"Rodin did not return a geometry file. Downloaded: {downloaded_files}")

    return RodinResult(
        task_uuid=task_uuid,
        subscription_key=subscription_key,
        glb_path=asset_path,
        preview_path=preview_path,
        downloaded_files=downloaded_files,
    )


def post_json(endpoint: str, **kwargs) -> dict:
    response = requests.post(f"{RODIN_API_BASE_URL}{endpoint}", **kwargs)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text[:1000]
        raise RodinError(f"Rodin request failed with HTTP {response.status_code}: {detail}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise RodinError(f"Rodin returned non-JSON response: {response.text[:1000]}") from exc


def download_file(url: str, output_dir: str, name: str, index: int) -> str:
    safe_name = sanitize_filename(name)
    _, ext = os.path.splitext(safe_name)
    if not ext:
        url_ext = os.path.splitext(url.split("?", 1)[0])[1]
        safe_name += url_ext or ".bin"

    path = os.path.join(output_dir, f"rodin_{index}_{safe_name}")
    with requests.get(url, stream=True, timeout=RODIN_DOWNLOAD_TIMEOUT) as response:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:1000]
            raise RodinError(f"Rodin file download failed with HTTP {response.status_code}: {detail}") from exc
        with open(path, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out_file.write(chunk)
    return path


def sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "rodin_result"


def update_progress(progress: Optional[Callable], value: float, desc: str) -> None:
    if progress is None:
        return
    try:
        progress(value, desc=desc)
    except TypeError:
        progress(value)
