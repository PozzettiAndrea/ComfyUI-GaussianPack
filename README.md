
https://github.com/user-attachments/assets/582f6b84-0a83-4b8a-b449-2cf9bf184bc5

https://github.com/user-attachments/assets/74f17146-3e5b-4dcf-8ab1-dd2e9d18e474

> [!WARNING]
> Warning, uses experimental package `comfy-env` to attempt a one click isolated install. Will download and use pixi package manager.

# ComfyUI-GaussianPack

## Installation

Three options, in order of speed → reliability:

1. **ComfyUI Manager (recommended)** — search for `GaussianPack` in the Manager and click Install from the highest version displayed. If that doesn't work, try nightly.
2. **Manager via Git URL** — in ComfyUI Manager: "Install via Git URL" with `https://github.com/PozzettiAndrea/ComfyUI-GaussianPack.git`.
3. **Manual (most reliable)**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/PozzettiAndrea/ComfyUI-GaussianPack.git
   cd ComfyUI-GaussianPack
   pip install -r requirements.txt --upgrade
   python install.py
   ```

> **Please report any problems** you hit during installation or use of my nodes — open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-GaussianPack/discussions) or [Issue](https://github.com/PozzettiAndrea/ComfyUI-GaussianPack/issues). Very grateful for your help! 🙏

---


<div align="center">
<a href="https://pozzettiandrea.github.io/ComfyUI-GaussianPack/">
<img src="https://raw.githubusercontent.com/PozzettiAndrea/ComfyUI-GaussianPack/dev/assets/gallery-preview.png" alt="Workflow Test Gallery" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/ComfyUI-GaussianPack/">View Live Test Gallery →</a></b>
</div>

https://github.com/user-attachments/assets/4ef6b921-52a0-4483-8efd-627d62c7e207

Tiny Gaussian-splat viewer for ComfyUI. One node, `Preview Gaussians`, with a
gsplat.js WebGL preview and camera-intrinsics widgets (FOV, image width, image height).

```
ply_path -- Preview Gaussians
```

GPL-3.0. Forked structurally from
[XuanYu-github/comfyui-PlyPreview](https://github.com/XuanYu-github/comfyui-PlyPreview).
