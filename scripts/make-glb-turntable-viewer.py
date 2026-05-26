#!/usr/bin/env python3
import argparse
import base64
import html
import os
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #111418;
      color: #eef2f5;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #viewport {{
      width: 100vw;
      height: 100vh;
      display: block;
    }}
    #toolbar {{
      position: fixed;
      left: 16px;
      top: 16px;
      display: flex;
      gap: 8px;
      align-items: center;
      background: rgba(10, 12, 14, 0.72);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 8px;
      padding: 10px;
      backdrop-filter: blur(8px);
    }}
    button, a {{
      appearance: none;
      border: 1px solid rgba(255, 255, 255, 0.2);
      background: #f4f7fb;
      color: #121417;
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      text-decoration: none;
      cursor: pointer;
    }}
    #status {{
      opacity: 0.82;
      font-size: 13px;
      max-width: 52vw;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
  </style>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://unpkg.com/three@0.164.1/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.164.1/examples/jsm/"
      }}
    }}
  </script>
</head>
<body>
  <canvas id="viewport"></canvas>
  <div id="toolbar">
    <button id="toggle">Pause</button>
    <button id="record">Record {record_seconds}s</button>
    <a id="download" href="{data_uri}" download="{download_name}">Download GLB</a>
    <span id="status">{status}</span>
  </div>
  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
    import {{ GLTFLoader }} from "three/addons/loaders/GLTFLoader.js";

    const canvas = document.getElementById("viewport");
    const status = document.getElementById("status");
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, preserveDrawingBuffer: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111418);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const hemi = new THREE.HemisphereLight(0xffffff, 0x283040, 2.2);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 3.4);
    key.position.set(4, 7, 6);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x9fb8ff, 1.2);
    fill.position.set(-5, 3, -4);
    scene.add(fill);

    const root = new THREE.Group();
    scene.add(root);
    let rotate = true;

    function resize() {{
      const width = window.innerWidth;
      const height = window.innerHeight;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }}
    window.addEventListener("resize", resize);
    resize();

    const loader = new GLTFLoader();
    const dataUri = "{data_uri}";
    fetch(dataUri)
      .then((response) => response.arrayBuffer())
      .then((buffer) => {{
        loader.parse(buffer, "", (gltf) => {{
          root.add(gltf.scene);
          const box = new THREE.Box3().setFromObject(root);
          const size = box.getSize(new THREE.Vector3());
          const center = box.getCenter(new THREE.Vector3());
          root.position.sub(center);
          const radius = Math.max(size.x, size.y, size.z) || 1;
          camera.position.set(radius * 1.9, radius * 1.15, radius * 2.4);
          camera.near = Math.max(0.001, radius / 1000);
          camera.far = radius * 100;
          controls.target.set(0, 0, 0);
          camera.updateProjectionMatrix();
          status.textContent = "{status}";
        }}, (error) => {{
          status.textContent = "GLB load failed: " + error.message;
          console.error(error);
        }});
      }});

    document.getElementById("toggle").addEventListener("click", (event) => {{
      rotate = !rotate;
      event.currentTarget.textContent = rotate ? "Pause" : "Rotate";
    }});

    document.getElementById("record").addEventListener("click", async (event) => {{
      if (!window.MediaRecorder) {{
        status.textContent = "MediaRecorder is not available in this browser.";
        return;
      }}
      event.currentTarget.disabled = true;
      const chunks = [];
      const stream = renderer.domElement.captureStream(30);
      const recorder = new MediaRecorder(stream, {{ mimeType: "video/webm" }});
      recorder.ondataavailable = (e) => {{
        if (e.data.size) chunks.push(e.data);
      }};
      recorder.onstop = () => {{
        const blob = new Blob(chunks, {{ type: "video/webm" }});
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "{video_name}";
        link.click();
        URL.revokeObjectURL(url);
        event.currentTarget.disabled = false;
        status.textContent = "Recording saved.";
      }};
      status.textContent = "Recording...";
      recorder.start();
      setTimeout(() => recorder.stop(), {record_ms});
    }});

    function animate() {{
      requestAnimationFrame(animate);
      if (rotate) root.rotation.y += 0.01;
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a standalone rotating GLB viewer with browser-side video recording.")
    parser.add_argument("glb", help="Path to a .glb file.")
    parser.add_argument("-o", "--output", help="Output HTML path. Defaults to <glb stem>-turntable.html.")
    parser.add_argument("--title", help="Viewer title. Defaults to the GLB filename.")
    parser.add_argument("--record-seconds", type=int, default=8, help="Duration for the Record button.")
    args = parser.parse_args()

    glb_path = Path(args.glb).expanduser().resolve()
    if not glb_path.exists():
        raise SystemExit(f"GLB file does not exist: {glb_path}")
    if glb_path.suffix.lower() != ".glb":
        raise SystemExit("This viewer expects a .glb file.")

    output_path = Path(args.output).expanduser().resolve() if args.output else glb_path.with_name(f"{glb_path.stem}-turntable.html")
    encoded = base64.b64encode(glb_path.read_bytes()).decode("ascii")
    data_uri = f"data:model/gltf-binary;base64,{encoded}"
    size_mb = os.path.getsize(glb_path) / (1024 * 1024)
    title = args.title or glb_path.name
    html_text = HTML_TEMPLATE.format(
        title=html.escape(title),
        data_uri=data_uri,
        download_name=html.escape(glb_path.name),
        video_name=html.escape(f"{glb_path.stem}-turntable.webm"),
        record_seconds=max(1, args.record_seconds),
        record_ms=max(1, args.record_seconds) * 1000,
        status=html.escape(f"{glb_path.name} ({size_mb:.1f} MB)"),
    )
    output_path.write_text(html_text, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
