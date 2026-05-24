import gradio as gr

import os
import argparse
import re
import subprocess
import tempfile
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datetime import datetime
import shutil
import cv2
from typing import *
import torch
import numpy as np
from PIL import Image
import base64
import io
from hitem3d_api import (
    Hitem3DError,
    generate_images_to_3d as generate_hitem3d_images_to_3d,
    has_hitem3d_credentials,
)
from local_mesh_repair import repair_mesh_locally
from rodin_api import (
    RodinError,
    generate_image_to_3d as generate_rodin_image_to_3d,
    generate_images_to_3d as generate_rodin_images_to_3d,
    get_rodin_api_key,
)
from threedai_api import (
    ThreeDAIStudioError,
    generate_tripo_p1,
    get_threedai_api_key,
    repair_mesh as repair_threedai_mesh,
)


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
TRELLIS_BACKEND = "TRELLIS.2 (local)"
RODIN_BACKEND = "Rodin Gen-2 (Hyper3D API)"
RODIN25_BACKEND = "Rodin Gen-2.5 (Hyper3D API)"
HITEM3D_BACKEND = "Hitem3D API"
TRIPO_P1_BACKEND = "Tripo P1 (3D AI Studio API)"
HITEM3D_MODELS = {
    "Portrait v2.1": "scene-portraitv2.1",
    "General v2.1": "hitem3dv2.1",
}
RODIN25_TIERS = [
    "Gen-2.5-Extreme-Low",
    "Gen-2.5-Low",
    "Gen-2.5-Medium",
    "Gen-2.5-High",
    "Gen-2.5-Extreme-High",
]
RODIN_GEOMETRY_FORMATS = ["glb", "usdz", "fbx", "obj", "stl"]
RODIN_MATERIALS = ["PBR", "Shaded", "All", "None"]
RODIN_TEXTURE_MODES = ["legacy", "extreme-low", "low", "medium", "high"]
RODIN_GEOMETRY_INSTRUCT_MODES = ["faithful", "creative"]
TRIPO_TEXTURE_QUALITIES = ["standard", "detailed"]
TRIPO_TEXTURE_ALIGNMENTS = ["original_image", "geometry"]
TRIPO_ORIENTATIONS = ["default", "align_image"]
TRIPO_REPAIR_QUALITIES = ["low", "default", "max"]
TRIPO_REPAIR_TOPOLOGIES = ["tris", "quads"]
SNAPSHOT_TAB = "snapshots"
GLB_TAB = "interactive-glb"
MODEL3D_SUPPORTED_EXTS = (".glb", ".gltf", ".obj", ".stl")
MODES = [
    {"name": "Normal", "icon": "assets/app/normal.png", "render_key": "normal"},
    {"name": "Clay render", "icon": "assets/app/clay.png", "render_key": "clay"},
    {"name": "Base color", "icon": "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "HDRI forest", "icon": "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "HDRI sunset", "icon": "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "HDRI courtyard", "icon": "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]
STEPS = 8
DEFAULT_MODE = 3
DEFAULT_STEP = 3
pipeline = None
envmap = None
rembg_model = None


def ensure_ffmpeg_on_path() -> None:
    """Expose imageio-ffmpeg's bundled binary for Gradio video preprocessing."""
    if shutil.which("ffmpeg"):
        return

    try:
        import imageio_ffmpeg
    except Exception:
        return

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not os.path.exists(ffmpeg_exe):
        return

    shim_dir = os.path.join("/tmp", "trellis2-bin")
    shim_path = os.path.join(shim_dir, "ffmpeg")
    os.makedirs(shim_dir, exist_ok=True)
    if not os.path.exists(shim_path):
        try:
            os.symlink(ffmpeg_exe, shim_path)
        except FileExistsError:
            pass
    os.environ["PATH"] = f"{shim_dir}:{os.environ.get('PATH', '')}"


ensure_ffmpeg_on_path()


css = """
/* Overwrite Gradio Default Style */
.stepper-wrapper {
    padding: 0;
}

.stepper-container {
    padding: 0;
    align-items: center;
}

.step-button {
    flex-direction: row;
}

.step-connector {
    transform: none;
}

.step-number {
    width: 16px;
    height: 16px;
}

.step-label {
    position: relative;
    bottom: 0;
}

.wrap.center.full {
    inset: 0;
    height: 100%;
}

.wrap.center.full.translucent {
    background: var(--block-background-fill);
}

.meta-text-center {
    display: block !important;
    position: absolute !important;
    top: unset !important;
    bottom: 0 !important;
    right: 0 !important;
    transform: unset !important;
}

/* Previewer */
.previewer-container {
    position: relative;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    width: 100%;
    height: 722px;
    margin: 0 auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.previewer-container .tips-icon {
    position: absolute;
    right: 10px;
    top: 10px;
    z-index: 10;
    border-radius: 10px;
    color: #fff;
    background-color: var(--color-accent);
    padding: 3px 6px;
    user-select: none;
}

.previewer-container .tips-text {
    position: absolute;
    right: 10px;
    top: 50px;
    color: #fff;
    background-color: var(--color-accent);
    border-radius: 10px;
    padding: 6px;
    text-align: left;
    max-width: 300px;
    z-index: 10;
    transition: all 0.3s;
    opacity: 0%;
    user-select: none;
}

.previewer-container .tips-text p {
    font-size: 14px;
    line-height: 1.2;
}

.tips-icon:hover + .tips-text { 
    display: block;
    opacity: 100%;
}

/* Row 1: Display Modes */
.previewer-container .mode-row {
    width: 100%;
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.previewer-container .mode-btn {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    cursor: pointer;
    opacity: 0.5;
    transition: all 0.2s;
    border: 2px solid #ddd;
    object-fit: cover;
}
.previewer-container .mode-btn:hover { opacity: 0.9; transform: scale(1.1); }
.previewer-container .mode-btn.active {
    opacity: 1;
    border-color: var(--color-accent);
    transform: scale(1.1);
}

/* Row 2: Display Image */
.previewer-container .display-row {
    margin-bottom: 20px;
    min-height: 400px;
    width: 100%;
    flex-grow: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}
.previewer-container .previewer-main-image {
    max-width: 100%;
    max-height: 100%;
    flex-grow: 1;
    object-fit: contain;
    display: none;
}
.previewer-container .previewer-main-image.visible {
    display: block;
}

/* Row 3: Custom HTML Slider */
.previewer-container .slider-row {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 0 10px;
}

.previewer-container input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    max-width: 400px;
    background: transparent;
}
.previewer-container input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 8px;
    cursor: pointer;
    background: #ddd;
    border-radius: 5px;
}
.previewer-container input[type=range]::-webkit-slider-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: var(--color-accent);
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -6px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: transform 0.1s;
}
.previewer-container input[type=range]::-webkit-slider-thumb:hover {
    transform: scale(1.2);
}

/* Overwrite Previewer Block Style */
.gradio-container .padded:has(.previewer-container) {
    padding: 0 !important;
}

.gradio-container:has(.previewer-container) [data-testid="block-label"] {
    position: absolute;
    top: 0;
    left: 0;
}
"""


head = """
<script>
    function refreshView(mode, step) {
        // 1. Find current mode and step
        const allImgs = document.querySelectorAll('.previewer-main-image');
        for (let i = 0; i < allImgs.length; i++) {
            const img = allImgs[i];
            if (img.classList.contains('visible')) {
                const id = img.id;
                const [_, m, s] = id.split('-');
                if (mode === -1) mode = parseInt(m.slice(1));
                if (step === -1) step = parseInt(s.slice(1));
                break;
            }
        }
        
        // 2. Hide ALL images
        // We select all elements with class 'previewer-main-image'
        allImgs.forEach(img => img.classList.remove('visible'));

        // 3. Construct the specific ID for the current state
        // Format: view-m{mode}-s{step}
        const targetId = 'view-m' + mode + '-s' + step;
        const targetImg = document.getElementById(targetId);

        // 4. Show ONLY the target
        if (targetImg) {
            targetImg.classList.add('visible');
        }

        // 5. Update Button Highlights
        const allBtns = document.querySelectorAll('.mode-btn');
        allBtns.forEach((btn, idx) => {
            if (idx === mode) btn.classList.add('active');
            else btn.classList.remove('active');
        });
    }
    
    // --- Action: Switch Mode ---
    function selectMode(mode) {
        refreshView(mode, -1);
    }
    
    // --- Action: Slider Change ---
    function onSliderChange(val) {
        refreshView(-1, parseInt(val));
    }
</script>
"""


empty_html = f"""
<div class="previewer-container">
    <svg style=" opacity: .5; height: var(--size-5); color: var(--body-text-color);"
    xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
</div>
"""


def get_trellis_pipeline():
    global pipeline, envmap
    if pipeline is None:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
        pipeline.cuda()

    if envmap is None:
        from trellis2.renderers import EnvMap

        envmap = {
            'forest': EnvMap(torch.tensor(
                cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            )),
            'sunset': EnvMap(torch.tensor(
                cv2.cvtColor(cv2.imread('assets/hdri/sunset.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            )),
            'courtyard': EnvMap(torch.tensor(
                cv2.cvtColor(cv2.imread('assets/hdri/courtyard.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            )),
        }

    return pipeline, envmap


def get_rembg_model():
    global rembg_model
    if pipeline is not None and getattr(pipeline, "rembg_model", None) is not None:
        return pipeline.rembg_model

    if rembg_model is None:
        from trellis2.pipelines.rembg import BiRefNet

        rembg_model = BiRefNet()
    return rembg_model


def image_to_base64(image):
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="jpeg", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir, ignore_errors=True)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    pipeline, _ = get_trellis_pipeline()
    processed_image = pipeline.preprocess_image(image)
    return processed_image


def has_alpha_mask(image: Image.Image) -> bool:
    if image.mode != 'RGBA':
        return False
    alpha = np.array(image)[:, :, 3]
    return not np.all(alpha == 255)


def crop_to_alpha(image: Image.Image) -> Image.Image:
    image = image.convert('RGBA')
    alpha = np.array(image)[:, :, 3]
    bbox_pixels = np.argwhere(alpha > 0.8 * 255)
    if len(bbox_pixels) == 0:
        return image

    left = int(np.min(bbox_pixels[:, 1]))
    top = int(np.min(bbox_pixels[:, 0]))
    right = int(np.max(bbox_pixels[:, 1])) + 1
    bottom = int(np.max(bbox_pixels[:, 0])) + 1
    center = (left + right) / 2, (top + bottom) / 2
    size = max(right - left, bottom - top)
    bbox = (
        int(center[0] - size // 2),
        int(center[1] - size // 2),
        int(center[0] + (size + 1) // 2),
        int(center[1] + (size + 1) // 2),
    )
    return image.crop(bbox)


def prepare_rodin_image(
    image: Image.Image,
    remove_background: bool,
    progress=gr.Progress(track_tqdm=True),
) -> Image.Image:
    if not remove_background:
        return image

    max_size = max(image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)

    if has_alpha_mask(image):
        return crop_to_alpha(image)

    if progress is not None:
        try:
            progress(0.01, desc="Removing background")
        except TypeError:
            progress(0.01)

    model = get_rembg_model()
    model.to("cuda")
    try:
        output = model(image.convert('RGB'))
    finally:
        if pipeline is None and hasattr(model, "cpu"):
            model.cpu()
    return crop_to_alpha(output)


def save_api_image(image: Image.Image, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    return path


def get_video_path(video: Any) -> Optional[str]:
    if video is None:
        return None
    if isinstance(video, str):
        return video
    if isinstance(video, (list, tuple)):
        for value in video:
            if isinstance(value, str):
                return value
    if isinstance(video, dict):
        for key in ("video", "path", "name"):
            value = video.get(key)
            if isinstance(value, str):
                return value
    return None


def get_uploaded_file_path(file_value: Any) -> Optional[str]:
    if file_value is None:
        return None
    if isinstance(file_value, str):
        return file_value
    if isinstance(file_value, dict):
        for key in ("path", "name"):
            value = file_value.get(key)
            if isinstance(value, str):
                return value
    for attr in ("path", "name"):
        value = getattr(file_value, attr, None)
        if isinstance(value, str):
            return value
    return None


def get_uploaded_file_paths(file_values: Any) -> list[str]:
    if file_values is None:
        return []
    if isinstance(file_values, (list, tuple)):
        paths = []
        for item in file_values:
            paths.extend(get_uploaded_file_paths(item))
        return dedupe_paths(paths)
    path = get_uploaded_file_path(file_values)
    return [path] if path else []


def dedupe_paths(paths: list[str]) -> list[str]:
    unique_paths = []
    seen = set()
    for path in paths:
        canonical = os.path.realpath(path) if os.path.exists(path) else os.path.abspath(path)
        if canonical in seen:
            continue
        seen.add(canonical)
        unique_paths.append(path)
    return unique_paths


def safe_upload_filename(video_path: str) -> str:
    name = os.path.basename(video_path) or "video"
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "video"
    ext = re.sub(r"[^A-Za-z0-9.]+", "", ext)[:12] or ".mp4"
    parent = os.path.basename(os.path.dirname(video_path))
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", parent)[:12] or "upload"
    return f"{prefix}_{stem}{ext}"


def persist_hitem_video_upload(video: Any) -> Optional[str]:
    video_path = get_video_path(video)
    if not video_path or not os.path.exists(video_path):
        return None

    upload_dir = os.path.join(TMP_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, safe_upload_filename(video_path))
    if os.path.abspath(video_path) == os.path.abspath(saved_path):
        return saved_path

    if not os.path.exists(saved_path) or os.path.getsize(saved_path) != os.path.getsize(video_path):
        shutil.copy2(video_path, saved_path)
    return saved_path


def get_preferred_video_path(video: Any, saved_video_path: Optional[str] = None) -> Optional[str]:
    if saved_video_path and os.path.exists(saved_video_path):
        return saved_video_path
    return get_video_path(video)


def get_video_duration(video_path: str) -> float:
    ensure_ffmpeg_on_path()
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        return 0.0

    result = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-i", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        return 0.0

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clamp_video_time(value: float, duration: float) -> float:
    if duration <= 0:
        return max(0.0, float(value or 0.0))
    return min(max(0.0, float(value or 0.0)), max(0.0, duration - 0.001))


def extract_frame_with_ffmpeg(video_path: str, target_time: float, output_path: str) -> Image.Image:
    ensure_ffmpeg_on_path()
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise gr.Error("ffmpeg is not available for video frame extraction.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        os.remove(output_path)
    except FileNotFoundError:
        pass
    result = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{target_time:.3f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            output_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        detail = result.stderr.strip()
        suffix = f" ffmpeg said: {detail[:500]}" if detail else ""
        raise gr.Error(f"Could not read video frame at {target_time:.2f}s.{suffix}")

    with Image.open(output_path) as image:
        return image.convert("RGB")


def preview_hitem_video_frame(video: Any, base_time: float, saved_video_path: str = "") -> tuple[Optional[Image.Image], str]:
    video_path = get_preferred_video_path(video, saved_video_path)
    if not video_path:
        return None, ""
    if not os.path.exists(video_path):
        return None, f"Video file is missing: {video_path}"

    try:
        duration = get_video_duration(video_path)
        target_time = clamp_video_time(base_time, duration)
        with tempfile.TemporaryDirectory(prefix="trellis2-video-preview-") as preview_dir:
            preview_path = os.path.join(preview_dir, "base_frame.png")
            image = extract_frame_with_ffmpeg(video_path, target_time, preview_path)
    except Exception as exc:
        return None, str(exc)

    if duration > 0:
        return image, f"Base frame: {target_time:.2f}s / {duration:.2f}s"
    return image, f"Base frame: {target_time:.2f}s"


def update_hitem_video_controls(video: Any) -> tuple[Any, Optional[Image.Image], str, str]:
    try:
        saved_video_path = persist_hitem_video_upload(video)
    except Exception as exc:
        return gr.update(value=0.0, maximum=60.0), None, f"Could not save uploaded video: {exc}", ""

    video_path = get_preferred_video_path(video, saved_video_path)
    if not video_path or not os.path.exists(video_path):
        return gr.update(value=0.0, maximum=60.0), None, "", ""

    duration = get_video_duration(video_path)
    maximum = max(0.1, duration) if duration > 0 else 60.0
    preview, status = preview_hitem_video_frame(video, 0.0, saved_video_path or "")
    if saved_video_path:
        status = f"{status} | Saved: {saved_video_path}"
    return gr.update(value=0.0, maximum=maximum), preview, status, saved_video_path or ""


def extract_video_frames(
    video_path: str,
    base_time: float,
    frame_count: int,
    spacing: float,
    output_dir: str,
    remove_background: bool,
    *,
    max_frames: int = 4,
    prefix: str = "video",
    progress=gr.Progress(track_tqdm=True),
) -> list[str]:
    if not os.path.exists(video_path):
        raise gr.Error(f"Video file is missing: {video_path}")

    duration = get_video_duration(video_path)
    base_time = clamp_video_time(base_time, duration)
    frame_count = int(max(1, min(max_frames, frame_count)))
    spacing = max(0.0, spacing)
    offsets = (np.arange(frame_count) - (frame_count - 1) / 2) * spacing
    paths = []
    for index, offset in enumerate(offsets, start=1):
        target_time = max(0.0, base_time + float(offset))
        if duration > 0:
            target_time = min(target_time, max(0.0, duration - 0.001))
        raw_path = os.path.join(output_dir, f"{prefix}_raw_{index:02d}.png")
        image = extract_frame_with_ffmpeg(video_path, target_time, raw_path)
        prepared = prepare_rodin_image(image, remove_background, progress=progress)
        path = os.path.join(output_dir, f"{prefix}_frame_{index:02d}.png")
        paths.append(save_api_image(prepared, path))
        if raw_path != path:
            try:
                os.remove(raw_path)
            except OSError:
                pass
    return paths


def prepare_hitem3d_inputs(
    image: Optional[Image.Image],
    video: Any,
    saved_video_path: str,
    use_video: bool,
    base_time: float,
    frame_count: int,
    spacing: float,
    remove_background: bool,
    user_dir: str,
    progress=gr.Progress(track_tqdm=True),
) -> list[str]:
    if use_video:
        video_path = get_preferred_video_path(video, saved_video_path)
        if not video_path:
            raise gr.Error("Upload a video or turn off Use Video Frames.")
        return extract_video_frames(
            video_path,
            base_time,
            frame_count,
            spacing,
            user_dir,
            remove_background,
            max_frames=4,
            prefix="hitem3d_video",
            progress=progress,
        )

    if image is None:
        raise gr.Error("Upload an image first.")
    prepared = prepare_rodin_image(image, remove_background, progress=progress)
    return [save_api_image(prepared, os.path.join(user_dir, "hitem3d_input_01.png"))]


def prepare_rodin_inputs(
    image: Optional[Image.Image],
    video: Any,
    saved_video_path: str,
    use_video: bool,
    base_time: float,
    frame_count: int,
    spacing: float,
    remove_background: bool,
    user_dir: str,
    progress=gr.Progress(track_tqdm=True),
) -> list[str]:
    if use_video:
        video_path = get_preferred_video_path(video, saved_video_path)
        if not video_path:
            raise gr.Error("Upload a video or turn off Use Video Frames.")
        return extract_video_frames(
            video_path,
            base_time,
            frame_count,
            spacing,
            user_dir,
            remove_background,
            max_frames=5,
            prefix="rodin_video",
            progress=progress,
        )

    if image is None:
        return []
    prepared = prepare_rodin_image(image, remove_background, progress=progress)
    return [save_api_image(prepared, os.path.join(user_dir, "rodin_input_01.png"))]


def prepare_tripo_inputs(
    image: Optional[Image.Image],
    multiview_images: Any,
    video: Any,
    saved_video_path: str,
    use_video: bool,
    base_time: float,
    frame_count: int,
    spacing: float,
    remove_background: bool,
    user_dir: str,
    progress=gr.Progress(track_tqdm=True),
) -> list[str]:
    if use_video:
        video_path = get_preferred_video_path(video, saved_video_path)
        if not video_path:
            raise gr.Error("Upload a video or turn off Use Video Frames.")
        return extract_video_frames(
            video_path,
            base_time,
            frame_count,
            spacing,
            user_dir,
            remove_background,
            max_frames=4,
            prefix="tripo_video",
            progress=progress,
        )

    uploaded_file_paths = get_uploaded_file_paths(multiview_images)
    missing_upload_paths = [path for path in uploaded_file_paths if not os.path.exists(path)]
    if missing_upload_paths:
        raise gr.Error("One or more uploaded multiview images are no longer available. Re-upload the multiview images.")

    upload_paths = uploaded_file_paths
    if len(upload_paths) > 4:
        raise gr.Error(f"Tripo P1 supports at most 4 multiview images; got {len(upload_paths)} distinct files.")

    use_main_image = image is not None and len(upload_paths) < 4
    if use_main_image:
        prepared = prepare_rodin_image(image, remove_background, progress=progress)
        paths = [save_api_image(prepared, os.path.join(user_dir, "tripo_input_01.png"))]
    else:
        paths = []

    for source_path in upload_paths:
        with Image.open(source_path) as uploaded:
            prepared = prepare_rodin_image(uploaded.convert("RGBA"), remove_background, progress=progress)
        index = len(paths) + 1
        paths.append(save_api_image(prepared, os.path.join(user_dir, f"tripo_input_{index:02d}.png")))

    return paths


def parse_optional_int(enabled: bool, value: Any, label: str = "Value") -> Optional[int]:
    if not enabled:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise gr.Error(f"{label} must be an integer.")


def model3d_path_or_none(path: Optional[str]) -> Optional[str]:
    if path and path.lower().endswith(MODEL3D_SUPPORTED_EXTS):
        return path
    return None


def preview_tab_for_model(path: Optional[str]) -> Any:
    return gr.update(selected=GLB_TAB if model3d_path_or_none(path) else SNAPSHOT_TAB)


def hitem3d_resolution(model_label: str, speed_label: str) -> str:
    if speed_label == "Pro":
        return "1536pro"
    if model_label == "Portrait v2.1":
        return "1536profast"
    return "1536fast"


def pack_state(latents: tuple) -> dict:
    shape_slat, tex_slat, res = latents
    return {
        'backend': 'trellis',
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }
    
    
def unpack_state(state: dict) -> tuple:
    from trellis2.modules.sparse import SparseTensor

    shape_slat = SparseTensor(
        feats=torch.from_numpy(state['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(state['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, state['res']


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def rodin_result_html(preview_path: Optional[str], task_uuid: str) -> str:
    if preview_path:
        try:
            preview_image = Image.open(preview_path)
            preview_src = image_to_base64(preview_image)
            preview_markup = f"""
                <img class="previewer-main-image visible"
                     src="{preview_src}"
                     loading="eager"
                     style="display:block; max-height:100%; object-fit:contain;">
            """
        except Exception:
            preview_markup = f"<p>Rodin task {task_uuid} completed.</p>"
    else:
        preview_markup = f"<p>Rodin task {task_uuid} completed.</p>"

    return f"""
    <div class="previewer-container">
        <div class="display-row">
            {preview_markup}
        </div>
    </div>
    """


def hitem3d_result_html(cover_path: Optional[str], task_id: str) -> str:
    if cover_path:
        try:
            preview_image = Image.open(cover_path)
            preview_src = image_to_base64(preview_image)
            preview_markup = f"""
                <img class="previewer-main-image visible"
                     src="{preview_src}"
                     loading="eager"
                     style="display:block; max-height:100%; object-fit:contain;">
            """
        except Exception:
            preview_markup = f"<p>Hitem3D task {task_id} completed.</p>"
    else:
        preview_markup = f"<p>Hitem3D task {task_id} completed.</p>"

    return f"""
    <div class="previewer-container">
        <div class="display-row">
            {preview_markup}
        </div>
    </div>
    """


def api_result_html(provider: str, preview_path: Optional[str], task_id: str) -> str:
    if preview_path:
        try:
            preview_image = Image.open(preview_path)
            preview_src = image_to_base64(preview_image)
            preview_markup = f"""
                <img class="previewer-main-image visible"
                     src="{preview_src}"
                     loading="eager"
                     style="display:block; max-height:100%; object-fit:contain;">
            """
        except Exception:
            preview_markup = f"<p>{provider} task {task_id} completed.</p>"
    else:
        preview_markup = f"<p>{provider} task {task_id} completed.</p>"

    return f"""
    <div class="previewer-container">
        <div class="display-row">
            {preview_markup}
        </div>
    </div>
    """


def image_to_3d(
    image: Optional[Image.Image],
    tripo_multiview_images: Any,
    video: Any,
    hitem_video_saved_path: str,
    backend: str,
    seed: int,
    resolution: str,
    hitem_model: str,
    hitem_speed: str,
    hitem_face_count: int,
    hitem_remove_background: bool,
    hitem_use_video: bool,
    hitem_video_base_time: float,
    hitem_video_frame_count: int,
    hitem_video_frame_spacing: float,
    rodin_prompt: str,
    rodin_quality: str,
    rodin_mesh_mode: str,
    rodin_tapose: bool,
    rodin_remove_background: bool,
    rodin_use_original_alpha: bool,
    rodin_hd_texture: bool,
    rodin25_tier: str,
    rodin_geometry_file_format: str,
    rodin_material: str,
    rodin_use_quality_override: bool,
    rodin_quality_override: int,
    rodin_highpack: bool,
    rodin_preview_render: bool,
    rodin_texture_delight: bool,
    rodin_texture_mode: str,
    rodin_is_micro: bool,
    rodin_is_symmetric: bool,
    rodin_geometry_instruct_mode: str,
    rodin_bbox_condition: str,
    tripo_prompt: str,
    tripo_negative_prompt: str,
    tripo_face_limit: int,
    tripo_texture: bool,
    tripo_pbr: bool,
    tripo_texture_quality: str,
    tripo_remove_background: bool,
    tripo_use_model_seed: bool,
    tripo_model_seed: int,
    tripo_use_image_seed: bool,
    tripo_image_seed: int,
    tripo_use_texture_seed: bool,
    tripo_texture_seed: int,
    tripo_auto_size: bool,
    tripo_export_uv: bool,
    tripo_compress_geometry: bool,
    tripo_texture_alignment: str,
    tripo_orientation: str,
    tripo_enable_image_autofix: bool,
    tripo_local_repair_mesh: bool,
    tripo_repair_mesh: bool,
    tripo_repair_quality: str,
    tripo_repair_topology: str,
    tripo_repair_bake_textures: bool,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> tuple:
    if backend not in {HITEM3D_BACKEND, RODIN25_BACKEND, TRIPO_P1_BACKEND} and image is None:
        raise gr.Error("Upload an image first.")

    if backend == TRIPO_P1_BACKEND:
        api_key = get_threedai_api_key()
        if not api_key:
            raise gr.Error("Set THREEDAI_API_KEY, THREEDAISTUDIO_API_KEY, or AI3DSTUDIO_API_KEY before using Tripo P1.")

        user_dir = os.path.join(TMP_DIR, str(req.session_hash))
        tripo_dir = os.path.join(user_dir, "tripo_p1")
        image_paths = prepare_tripo_inputs(
            image,
            tripo_multiview_images,
            video,
            hitem_video_saved_path,
            hitem_use_video,
            hitem_video_base_time,
            hitem_video_frame_count,
            hitem_video_frame_spacing,
            tripo_remove_background,
            tripo_dir,
            progress=progress,
        )
        if not image_paths and not tripo_prompt.strip():
            raise gr.Error("Upload an image/video or provide a prompt for Tripo P1 text-to-3D.")

        try:
            result = generate_tripo_p1(
                image_paths,
                tripo_dir,
                api_key,
                prompt=tripo_prompt,
                negative_prompt=tripo_negative_prompt,
                face_limit=tripo_face_limit,
                texture=tripo_texture,
                pbr=tripo_pbr,
                texture_quality=tripo_texture_quality,
                model_seed=parse_optional_int(tripo_use_model_seed, tripo_model_seed, "Model Seed"),
                image_seed=parse_optional_int(tripo_use_image_seed, tripo_image_seed, "Image Seed"),
                texture_seed=parse_optional_int(tripo_use_texture_seed, tripo_texture_seed, "Texture Seed"),
                auto_size=tripo_auto_size,
                export_uv=tripo_export_uv,
                compress_geometry=tripo_compress_geometry,
                texture_alignment=tripo_texture_alignment,
                orientation=tripo_orientation,
                enable_image_autofix=tripo_enable_image_autofix,
                progress=progress,
            )
        except ThreeDAIStudioError as exc:
            raise gr.Error(str(exc)) from exc

        repair_task_id = None
        glb_path = result.glb_path
        asset_url = result.asset_url
        local_repair_status = None
        local_repair_satisfied = False
        if tripo_local_repair_mesh:
            local_repair = repair_mesh_locally(
                result.glb_path,
                tripo_dir,
                output_name="tripo_p1_local_repaired.glb",
            )
            local_repair_status = local_repair.details
            local_repair_satisfied = local_repair.watertight_after
            if local_repair.repaired:
                glb_path = local_repair.output_path

        needs_paid_repair = tripo_repair_mesh and not local_repair_satisfied
        if needs_paid_repair:
            try:
                repaired = repair_threedai_mesh(
                    result.glb_path,
                    tripo_dir,
                    api_key,
                    output_format="glb",
                    hollow=False,
                    topology=tripo_repair_topology,
                    quality=tripo_repair_quality,
                    bake_textures=tripo_repair_bake_textures,
                    progress=progress,
                )
            except ThreeDAIStudioError as exc:
                raise gr.Error(str(exc)) from exc
            repair_task_id = repaired.task_id
            glb_path = repaired.glb_path
            asset_url = repaired.asset_url

        state = {
            'backend': 'tripo_p1',
            'task_id': result.task_id,
            'repair_task_id': repair_task_id,
            'local_repair_status': local_repair_status,
            'glb_path': glb_path,
            'asset_url': asset_url,
            'input_paths': image_paths,
        }
        display_task_id = repair_task_id or result.task_id
        provider = "Tripo P1"
        if repair_task_id:
            provider = "Tripo P1 + Paid Mesh Repair"
        elif local_repair_status:
            provider = f"Tripo P1 + Local Mesh Repair ({local_repair_status})"
        return state, api_result_html(provider, None, display_task_id), glb_path, preview_tab_for_model(glb_path)

    if backend == HITEM3D_BACKEND:
        if not has_hitem3d_credentials():
            raise gr.Error("Set HITEM3D_CLIENT_ID and HITEM3D_CLIENT_SECRET before using Hitem3D.")

        user_dir = os.path.join(TMP_DIR, str(req.session_hash))
        hitem_dir = os.path.join(user_dir, "hitem3d")
        image_paths = prepare_hitem3d_inputs(
            image,
            video,
            hitem_video_saved_path,
            hitem_use_video,
            hitem_video_base_time,
            hitem_video_frame_count,
            hitem_video_frame_spacing,
            hitem_remove_background,
            hitem_dir,
            progress=progress,
        )
        try:
            result = generate_hitem3d_images_to_3d(
                image_paths,
                hitem_dir,
                model=HITEM3D_MODELS[hitem_model],
                resolution=hitem3d_resolution(hitem_model, hitem_speed),
                face_count=hitem_face_count,
                pbr=True,
                progress=progress,
            )
        except Hitem3DError as exc:
            raise gr.Error(str(exc)) from exc

        state = {
            'backend': 'hitem3d',
            'task_id': result.task_id,
            'glb_path': result.glb_path,
            'cover_path': result.cover_path,
            'input_paths': image_paths,
        }
        return state, hitem3d_result_html(result.cover_path, result.task_id), result.glb_path, preview_tab_for_model(result.glb_path)

    if backend == RODIN25_BACKEND:
        api_key = get_rodin_api_key()
        if not api_key:
            raise gr.Error("Set HYPER3D_API_KEY or RODIN_API_KEY before using Rodin.")

        user_dir = os.path.join(TMP_DIR, str(req.session_hash))
        rodin_dir = os.path.join(user_dir, "rodin25")
        image_paths = prepare_rodin_inputs(
            image,
            video,
            hitem_video_saved_path,
            hitem_use_video,
            hitem_video_base_time,
            hitem_video_frame_count,
            hitem_video_frame_spacing,
            rodin_remove_background,
            rodin_dir,
            progress=progress,
        )
        if not image_paths and not rodin_prompt.strip():
            raise gr.Error("Upload an image/video or provide a prompt for Rodin Gen-2.5 text-to-3D.")

        try:
            result = generate_rodin_images_to_3d(
                image_paths,
                rodin_dir,
                api_key,
                prompt=rodin_prompt,
                seed=seed,
                tier=rodin25_tier,
                quality=rodin_quality,
                mesh_mode=rodin_mesh_mode,
                tapose=rodin_tapose,
                use_original_alpha=rodin_use_original_alpha,
                geometry_file_format=rodin_geometry_file_format,
                material=rodin_material,
                quality_override=parse_optional_int(rodin_use_quality_override, rodin_quality_override, "Quality Override"),
                addons=["HighPack"] if rodin_highpack else [],
                preview_render=rodin_preview_render,
                hd_texture=rodin_hd_texture,
                texture_delight=rodin_texture_delight,
                texture_mode=rodin_texture_mode,
                is_micro=rodin_is_micro,
                is_symmetric=rodin_is_symmetric,
                geometry_instruct_mode=rodin_geometry_instruct_mode,
                bbox_condition=rodin_bbox_condition,
                progress=progress,
            )
        except RodinError as exc:
            raise gr.Error(str(exc)) from exc

        state = {
            'backend': 'rodin',
            'task_uuid': result.task_uuid,
            'subscription_key': result.subscription_key,
            'glb_path': result.glb_path,
            'preview_path': result.preview_path,
            'downloaded_files': result.downloaded_files,
            'input_paths': image_paths,
        }
        model3d_path = model3d_path_or_none(result.glb_path)
        return state, rodin_result_html(result.preview_path, result.task_uuid), model3d_path, preview_tab_for_model(model3d_path)

    if backend == RODIN_BACKEND:
        api_key = get_rodin_api_key()
        if not api_key:
            raise gr.Error("Set HYPER3D_API_KEY or RODIN_API_KEY before using Rodin.")

        user_dir = os.path.join(TMP_DIR, str(req.session_hash))
        rodin_image = prepare_rodin_image(
            image,
            rodin_remove_background,
            progress=progress,
        )
        try:
            result = generate_rodin_image_to_3d(
                rodin_image,
                user_dir,
                api_key,
                prompt=rodin_prompt,
                seed=seed,
                quality=rodin_quality,
                mesh_mode=rodin_mesh_mode,
                tapose=rodin_tapose,
                use_original_alpha=rodin_use_original_alpha,
                hd_texture=rodin_hd_texture,
                progress=progress,
            )
        except RodinError as exc:
            raise gr.Error(str(exc)) from exc

        state = {
            'backend': 'rodin',
            'task_uuid': result.task_uuid,
            'subscription_key': result.subscription_key,
            'glb_path': result.glb_path,
            'preview_path': result.preview_path,
            'downloaded_files': result.downloaded_files,
        }
        return state, rodin_result_html(result.preview_path, result.task_uuid), result.glb_path, preview_tab_for_model(result.glb_path)

    trellis_pipeline, trellis_envmap = get_trellis_pipeline()

    # --- Sampling ---
    outputs, latents = trellis_pipeline.run(
        image,
        seed=seed,
        preprocess_image=True,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
        },
        shape_slat_sampler_params={
            "steps": shape_slat_sampling_steps,
            "guidance_strength": shape_slat_guidance_strength,
            "guidance_rescale": shape_slat_guidance_rescale,
            "rescale_t": shape_slat_rescale_t,
        },
        tex_slat_sampler_params={
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
        },
        pipeline_type={
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[resolution],
        return_latent=True,
    )
    mesh = outputs[0]
    mesh.simplify(16777216) # nvdiffrast limit
    from trellis2.utils import render_utils

    images = render_utils.render_snapshot(mesh, resolution=1024, r=2, fov=36, nviews=STEPS, envmap=trellis_envmap)
    state = pack_state(latents)
    torch.cuda.empty_cache()
    
    # --- HTML Construction ---
    # The Stack of 48 Images
    images_html = ""
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(STEPS):
            # ID Naming Convention: view-m{mode}-s{step}
            unique_id = f"view-m{m_idx}-s{s_idx}"
            
            # Logic: Only Mode 0, Step 0 is visible initially
            is_visible = (m_idx == DEFAULT_MODE and s_idx == DEFAULT_STEP)
            vis_class = "visible" if is_visible else ""
            
            # Image Source
            img_base64 = image_to_base64(Image.fromarray(images[mode['render_key']][s_idx]))
            
            # Render the Tag
            images_html += f"""
                <img id="{unique_id}" 
                     class="previewer-main-image {vis_class}" 
                     src="{img_base64}" 
                     loading="eager">
            """
    
    # Button Row HTML
    btns_html = ""
    for idx, mode in enumerate(MODES):        
        active_class = "active" if idx == DEFAULT_MODE else ""
        # Note: onclick calls the JS function defined in Head
        btns_html += f"""
            <img src="{mode['icon_base64']}" 
                 class="mode-btn {active_class}" 
                 onclick="selectMode({idx})"
                 title="{mode['name']}">
        """
    
    # Assemble the full component
    full_html = f"""
    <div class="previewer-container">
        <div class="tips-wrapper">
            <div class="tips-icon">💡Tips</div>
            <div class="tips-text">
                <p>● <b>Render Mode</b> - Click on the circular buttons to switch between different render modes.</p>
                <p>● <b>View Angle</b> - Drag the slider to change the view angle.</p>
            </div>
        </div>
        
        <!-- Row 1: Viewport containing 48 static <img> tags -->
        <div class="display-row">
            {images_html}
        </div>
        
        <!-- Row 2 -->
        <div class="mode-row" id="btn-group">
            {btns_html}
        </div>

        <!-- Row 3: Slider -->
        <div class="slider-row">
            <input type="range" id="custom-slider" min="0" max="{STEPS - 1}" value="{DEFAULT_STEP}" step="1" oninput="onSliderChange(this.value)">
        </div>
    </div>
    """
    
    return state, full_html, None, gr.update(selected=SNAPSHOT_TAB)


def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        decimation_target (int): The target face count for decimation.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    if not state:
        raise gr.Error("Generate a 3D asset first.")

    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if state.get('backend') in {'rodin', 'hitem3d', 'tripo_p1'}:
        glb_path = state.get('glb_path')
        if not glb_path or not os.path.exists(glb_path):
            raise gr.Error("Generated asset is missing. Generate again.")
        model_path = model3d_path_or_none(glb_path)
        return model_path, model_path, glb_path

    trellis_pipeline, _ = get_trellis_pipeline()
    shape_slat, tex_slat, res = unpack_state(state)
    mesh = trellis_pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    import o_voxel

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=trellis_pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
    glb.export(glb_path, extension_webp=True)
    torch.cuda.empty_cache()
    return glb_path, glb_path, glb_path


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [TRELLIS.2](https://microsoft.github.io/TRELLIS.2), Rodin, Hitem3D, or Tripo P1
    * Upload an image or video source and click Generate to create a 3D asset.
    * Click Extract Asset to export and download the generated asset if you're satisfied with the result. Otherwise, try another time.
    """)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)
            hitem_video = gr.Video(label="Video Prompt", sources=["upload"], include_audio=True)
            backend = gr.Radio([TRELLIS_BACKEND, RODIN_BACKEND, RODIN25_BACKEND, HITEM3D_BACKEND, TRIPO_P1_BACKEND], label="Backend", value=TRELLIS_BACKEND)
            hitem_use_video = gr.Checkbox(label="Use Video Frames for API Backend", value=False)
            
            resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="1024")
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            decimation_target = gr.Slider(100000, 1000000, label="Decimation Target", value=500000, step=10000)
            texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)

            with gr.Accordion(label="Video Settings", open=False):
                hitem_video_base_time = gr.Slider(0.0, 60.0, label="Base Time (seconds)", value=0.0, step=0.05)
                hitem_video_frame_count = gr.Slider(1, 5, label="Frame Count", value=4, step=1)
                hitem_video_frame_spacing = gr.Slider(0.05, 1.0, label="Frame Spacing (seconds)", value=0.25, step=0.05)
                hitem_video_preview = gr.Image(label="Base Frame Preview", type="pil", height=240, interactive=False)
                hitem_video_status = gr.Textbox(label="Video Status", lines=1, interactive=False)

            with gr.Accordion(label="Hitem3D Settings", open=False):
                hitem_model = gr.Radio(list(HITEM3D_MODELS.keys()), label="Model", value="Portrait v2.1")
                hitem_speed = gr.Radio(["Fast", "Pro"], label="Resolution", value="Fast")
                hitem_face_count = gr.Slider(100000, 2000000, label="Face Count", value=800000, step=100000)
                hitem_remove_background = gr.Checkbox(label="Remove Background", value=True)

            with gr.Accordion(label="Rodin Settings", open=False):
                rodin_prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Optional image guidance prompt")
                rodin25_tier = gr.Radio(RODIN25_TIERS, label="Gen-2.5 Tier", value="Gen-2.5-High")
                rodin_quality = gr.Radio(["medium", "high", "low", "extra-low"], label="Quality", value="medium")
                rodin_mesh_mode = gr.Radio(["Raw", "Quad"], label="Mesh Mode", value="Raw")
                rodin_geometry_file_format = gr.Radio(RODIN_GEOMETRY_FORMATS, label="Geometry Format", value="glb")
                rodin_material = gr.Radio(RODIN_MATERIALS, label="Material", value="PBR")
                rodin_use_quality_override = gr.Checkbox(label="Use Quality Override", value=False)
                rodin_quality_override = gr.Slider(500, 2000000, label="Quality Override Faces", value=500000, step=500)
                rodin_tapose = gr.Checkbox(label="T/A Pose", value=False)
                rodin_remove_background = gr.Checkbox(label="Remove Background", value=True)
                rodin_use_original_alpha = gr.Checkbox(label="Use Original Alpha", value=True)
                rodin_highpack = gr.Checkbox(label="HighPack Addon", value=False)
                rodin_preview_render = gr.Checkbox(label="Preview Render", value=True)
                rodin_hd_texture = gr.Checkbox(label="HD Texture", value=False)
                rodin_texture_delight = gr.Checkbox(label="Texture Delight", value=False)
                rodin_texture_mode = gr.Radio(RODIN_TEXTURE_MODES, label="Texture Mode", value="high")
                rodin_is_micro = gr.Checkbox(label="Micro Detail", value=False)
                rodin_is_symmetric = gr.Checkbox(label="Symmetric", value=False)
                rodin_geometry_instruct_mode = gr.Radio(RODIN_GEOMETRY_INSTRUCT_MODES, label="Geometry Instruct Mode", value="faithful")
                rodin_bbox_condition = gr.Textbox(label="BBox Condition [Y,Z,X]", lines=1, placeholder="[50,80,50]")

            with gr.Accordion(label="Tripo P1 Settings", open=False):
                tripo_multiview_images = gr.File(label="Multiview Images (up to 4)", file_count="multiple", file_types=["image"], type="filepath")
                tripo_prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Required for text-to-3D")
                tripo_negative_prompt = gr.Textbox(label="Negative Prompt", lines=1)
                tripo_face_limit = gr.Slider(48, 20000, label="Face Limit", value=10000, step=1)
                tripo_texture = gr.Checkbox(label="Texture", value=True)
                tripo_pbr = gr.Checkbox(label="PBR", value=True)
                tripo_texture_quality = gr.Radio(TRIPO_TEXTURE_QUALITIES, label="Texture Quality", value="standard")
                tripo_remove_background = gr.Checkbox(label="Remove Background", value=True)
                tripo_use_model_seed = gr.Checkbox(label="Use Model Seed", value=False)
                tripo_model_seed = gr.Slider(0, MAX_SEED, label="Model Seed", value=0, step=1)
                tripo_use_image_seed = gr.Checkbox(label="Use Image Seed", value=False)
                tripo_image_seed = gr.Slider(0, MAX_SEED, label="Image Seed", value=0, step=1)
                tripo_use_texture_seed = gr.Checkbox(label="Use Texture Seed", value=False)
                tripo_texture_seed = gr.Slider(0, MAX_SEED, label="Texture Seed", value=0, step=1)
                tripo_auto_size = gr.Checkbox(label="Auto Size", value=False)
                tripo_export_uv = gr.Checkbox(label="Export UV", value=True)
                tripo_compress_geometry = gr.Checkbox(label="Compress Geometry", value=False)
                tripo_texture_alignment = gr.Radio(TRIPO_TEXTURE_ALIGNMENTS, label="Texture Alignment", value="original_image")
                tripo_orientation = gr.Radio(TRIPO_ORIENTATIONS, label="Orientation", value="align_image")
                tripo_enable_image_autofix = gr.Checkbox(label="Image Autofix", value=False)
                tripo_local_repair_mesh = gr.Checkbox(label="Try Local Mesh Repair (free)", value=False)
                tripo_repair_mesh = gr.Checkbox(label="Use Paid Mesh Repair If Needed (+60 credits)", value=False)
                tripo_repair_quality = gr.Radio(TRIPO_REPAIR_QUALITIES, label="Repair Quality", value="default")
                tripo_repair_topology = gr.Radio(TRIPO_REPAIR_TOPOLOGIES, label="Repair Topology", value="tris")
                tripo_repair_bake_textures = gr.Checkbox(label="Bake Textures During Repair", value=True)
            
            generate_btn = gr.Button("Generate")
                
            with gr.Accordion(label="Advanced Settings", open=False):                
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1)
                gr.Markdown("Stage 2: Shape Generation")
                with gr.Row():
                    shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01)
                    shape_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                gr.Markdown("Stage 3: Material Generation")
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                    tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)                

        with gr.Column(scale=10):
            with gr.Walkthrough(selected=0) as walkthrough:
                with gr.Step("Preview", id=0):
                    with gr.Tabs(selected=SNAPSHOT_TAB) as preview_tabs:
                        with gr.Tab("Rendered Snapshots", id=SNAPSHOT_TAB):
                            preview_output = gr.HTML(empty_html, label="3D Asset Preview", show_label=True, container=True)
                        with gr.Tab("Interactive GLB", id=GLB_TAB):
                            preview_glb_output = gr.Model3D(label="Interactive GLB Preview", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                    extract_btn = gr.Button("Extract Asset")
                with gr.Step("Extract", id=1):
                    glb_output = gr.Model3D(label="Extracted Asset", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                    download_btn = gr.DownloadButton(label="Download Asset")
                    
        with gr.Column(scale=1, min_width=172):
            examples = gr.Examples(
                examples=[
                    f'assets/example_image/{image}'
                    for image in os.listdir("assets/example_image")
                ],
                inputs=[image_prompt],
                examples_per_page=18,
            )
                    
    output_buf = gr.State()
    hitem_video_path_state = gr.State("")
    

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)

    hitem_video.change(
        update_hitem_video_controls,
        inputs=[hitem_video],
        outputs=[hitem_video_base_time, hitem_video_preview, hitem_video_status, hitem_video_path_state],
    )
    hitem_video_base_time.input(
        preview_hitem_video_frame,
        inputs=[hitem_video, hitem_video_base_time, hitem_video_path_state],
        outputs=[hitem_video_preview, hitem_video_status],
        queue=False,
    )
    
    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        lambda: gr.Walkthrough(selected=0), outputs=walkthrough
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, tripo_multiview_images, hitem_video, hitem_video_path_state, backend, seed, resolution,
            hitem_model, hitem_speed, hitem_face_count, hitem_remove_background,
            hitem_use_video, hitem_video_base_time, hitem_video_frame_count, hitem_video_frame_spacing,
            rodin_prompt, rodin_quality, rodin_mesh_mode, rodin_tapose, rodin_remove_background, rodin_use_original_alpha, rodin_hd_texture,
            rodin25_tier, rodin_geometry_file_format, rodin_material, rodin_use_quality_override, rodin_quality_override,
            rodin_highpack, rodin_preview_render, rodin_texture_delight, rodin_texture_mode, rodin_is_micro,
            rodin_is_symmetric, rodin_geometry_instruct_mode, rodin_bbox_condition,
            tripo_prompt, tripo_negative_prompt, tripo_face_limit, tripo_texture, tripo_pbr,
            tripo_texture_quality, tripo_remove_background, tripo_use_model_seed, tripo_model_seed,
            tripo_use_image_seed, tripo_image_seed, tripo_use_texture_seed, tripo_texture_seed,
            tripo_auto_size, tripo_export_uv, tripo_compress_geometry, tripo_texture_alignment,
            tripo_orientation, tripo_enable_image_autofix, tripo_local_repair_mesh, tripo_repair_mesh,
            tripo_repair_quality, tripo_repair_topology, tripo_repair_bake_textures,
            ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
            shape_slat_guidance_strength, shape_slat_guidance_rescale, shape_slat_sampling_steps, shape_slat_rescale_t,
            tex_slat_guidance_strength, tex_slat_guidance_rescale, tex_slat_sampling_steps, tex_slat_rescale_t,
        ],
        outputs=[output_buf, preview_output, preview_glb_output, preview_tabs],
    )
    
    extract_btn.click(
        lambda: gr.Walkthrough(selected=1), outputs=walkthrough
    ).then(
        extract_glb,
        inputs=[output_buf, decimation_target, texture_size],
        outputs=[preview_glb_output, glb_output, download_btn],
    )
        

# Launch the Gradio app
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TRELLIS.2 Gradio app.")
    parser.add_argument("--server-name", default=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--server-port", type=int, default=int(os.environ["GRADIO_SERVER_PORT"]) if os.environ.get("GRADIO_SERVER_PORT") else None)
    parser.add_argument("--share", action="store_true", default=os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true", "yes"})
    args = parser.parse_args()

    os.makedirs(TMP_DIR, exist_ok=True)

    # Construct ui components
    btn_img_base64_strs = {}
    for i in range(len(MODES)):
        icon = Image.open(MODES[i]['icon'])
        MODES[i]['icon_base64'] = image_to_base64(icon)
    
    demo.launch(css=css, head=head, server_name=args.server_name, server_port=args.server_port, share=args.share)
