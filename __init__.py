from comfy_env import register_nodes

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()

# Mount web/ under /extensions/ComfyUI-GaussianPack/* so the iframe-based
# viewers (PreviewGaussianSpectate, PreviewGaussians, PreviewGaussianCamera)
# can load their JS extension + viewer_gaussian.html. Without this, the
# browser hits 404 on the iframe src and no preview renders.
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]