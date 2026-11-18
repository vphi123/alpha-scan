"""Export scan.glb, alpha_scan.body_scan.v2 JSON, textures, deletion audit, and visual QA plots."""

from datetime import datetime, timezone
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import trimesh

    from .texturing import TexturedScanOutputs

from .config import (
    DEFAULT_MHR_VERSION,
    DEFAULT_SAM_CHECKPOINT_VERSION,
    PIPELINE_VERSION,
    SCHEMA_VERSION,
)
from .fusion import FusedSubjectParams


def build_body_scan_json(
    person_id: str,
    tier: str,
    sex: str,
    height_cm: float,
    weight_kg: float,
    circumferences_cm: Dict[str, float],
    loft: List[List[float]],
    fused: FusedSubjectParams,
    sam_version: str = DEFAULT_SAM_CHECKPOINT_VERSION,
    mhr_version: str = DEFAULT_MHR_VERSION,
    mesh_filename: str = "scan.glb",
    textures: Optional[Dict[str, Any]] = None,
    reconstruction_mode: Optional[str] = None,
    effective_reconstruction_mode: Optional[str] = None,
    proportion_audit: Optional[Dict[str, Any]] = None,
    core_mesh_filename: Optional[str] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble JSON payload conforming to alpha_scan.body_scan.v2 schema."""
    mesh_block: Dict[str, Any] = {
        "file": mesh_filename,
        "units": "m",
        "pose": "A",
    }
    if core_mesh_filename:
        mesh_block["core_file"] = core_mesh_filename

    payload: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "person_id": person_id,
        "tier": tier,
        "subject": {
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
        },
        "source": "sam3d_mhr",
        "model_versions": {
            "sam3d_body": sam_version,
            "mhr": mhr_version,
            "pipeline": PIPELINE_VERSION,
        },
        "circumferences_cm": circumferences_cm,
        "shape_profile": {"loft": loft},
        "quality": {
            "images_used": fused.images_used,
            "images_total": fused.images_total,
            "mean_confidence": round(fused.mean_confidence, 3),
            "confidence": "trend" if fused.images_used >= 3 else "low",
        },
        "mesh": mesh_block,
    }
    if textures is not None:
        payload["textures"] = textures
    if reconstruction_mode is not None:
        payload["reconstruction_mode"] = reconstruction_mode
    if effective_reconstruction_mode is not None:
        payload["effective_reconstruction_mode"] = effective_reconstruction_mode
    if proportion_audit is not None:
        payload["proportion_audit"] = proportion_audit
    if warnings is not None:
        payload["warnings"] = list(warnings)
    return payload


def export_scan_artifacts(
    full_mesh: "trimesh.Trimesh",
    body_json: Dict[str, Any],
    output_dir: str,
) -> Dict[str, str]:
    """Write scan.glb and body_scan.json to output_dir and return output file paths."""
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, body_json["mesh"]["file"])
    json_path = os.path.join(output_dir, "body_scan.json")

    full_mesh.export(glb_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(body_json, f, indent=2, ensure_ascii=False)

    return {"glb": glb_path, "json": json_path}


def build_texture_metadata(
    tex_outputs: "TexturedScanOutputs",
    privacy_level: int = 3,
    mesh_filename: str = "scan.glb",
    export_preview_glb: bool = True,
    views_used: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble privacy-scoped texture metadata block for body_scan.json."""
    levels: Dict[str, str] = {"1_plain": mesh_filename}
    previews: Dict[str, str] = {}
    if privacy_level >= 2:
        levels["2_blurred_face"] = "tex_blur.jpg"
        if export_preview_glb:
            previews["2_blurred_face"] = "scan_preview_blur.glb"
    if privacy_level >= 3:
        levels["3_clear_face"] = "tex_face.jpg"
        if export_preview_glb:
            previews["3_clear_face"] = "scan_preview_face.glb"

    res_h, res_w = tex_outputs.tex_face_rgb.shape[:2]
    res_px = int(max(res_h, res_w))
    meta: Dict[str, Any] = {
        "enabled": True,
        "privacy_level": int(privacy_level),
        "tex_size": res_px,
        "resolution": res_px,
        "uv_source": tex_outputs.uv_layout.source,
        "rasterizer": "pytorch3d" if tex_outputs.used_pytorch3d else "cpu_zbuf",
        "coverage_pct": round(float(getattr(tex_outputs, "coverage_pct", 0.0)), 2),
        "head_coverage_pct": round(float(getattr(tex_outputs, "head_coverage_pct", 0.0)), 2),
        "levels": levels,
    }
    if views_used is not None:
        meta["views_used"] = int(views_used)
    if privacy_level >= 2:
        meta["tex_blur"] = "tex_blur.jpg"
        if export_preview_glb:
            meta["preview_blur_glb"] = "scan_preview_blur.glb"
    if privacy_level >= 3:
        meta["tex_face"] = "tex_face.jpg"
        if export_preview_glb:
            meta["preview_face_glb"] = "scan_preview_face.glb"
    head_uv_layout = getattr(tex_outputs, "head_uv_layout", None)
    if head_uv_layout is not None:
        meta["materials"] = ["body", "head"]
        head_tex = getattr(tex_outputs, "head_tex_face_rgb", None)
        if head_tex is not None:
            h_res_h, h_res_w = head_tex.shape[:2]
            meta["head_resolution"] = int(max(h_res_h, h_res_w))
    if previews:
        meta["preview_glb"] = previews
    return meta


def _compute_smooth_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute smooth unit vertex normals across shared mesh topology with [0, 1, 0] degenerate fallback."""
    verts_f = np.asarray(verts, dtype=np.float32)
    faces_i = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(verts_f, dtype=np.float32)
    if len(faces_i) > 0:
        p0 = verts_f[faces_i[:, 0]]
        p1 = verts_f[faces_i[:, 1]]
        p2 = verts_f[faces_i[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)
        np.add.at(normals, faces_i[:, 0], fn)
        np.add.at(normals, faces_i[:, 1], fn)
        np.add.at(normals, faces_i[:, 2], fn)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = (lengths[:, 0] > 1e-8) & np.isfinite(lengths[:, 0])
    out = np.zeros_like(normals, dtype=np.float32)
    out[:, 1] = 1.0
    if np.any(valid):
        out[valid] = normals[valid] / lengths[valid]
    return out


def _opengl_uv_to_gltf(uv: np.ndarray) -> np.ndarray:
    """Convert OpenGL bottom-origin UV (v=0 bottom) to glTF 2.0 top-origin UV (v=0 top)."""
    uv_gltf = np.asarray(uv, dtype=np.float32).copy()
    uv_gltf[:, 1] = 1.0 - uv_gltf[:, 1]
    return uv_gltf


def _encode_rgb_jpeg_bytes(rgb_arr: np.ndarray, quality: int = 94) -> bytes:
    """Encode an RGB uint8 NumPy array into 4:4:4 JPEG bytes to avoid UV gutter chroma bleed."""
    import io

    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb_arr, dtype=np.uint8)).save(
        buf, format="JPEG", quality=quality, subsampling=0
    )
    return buf.getvalue()


def export_textured_scan_artifacts(
    verts_a: np.ndarray,
    body_json: Dict[str, Any],
    output_dir: str,
    tex_outputs: "TexturedScanOutputs",
    export_preview_glb: bool = True,
    privacy_level: int = 3,
    core_verts: Optional[np.ndarray] = None,
    core_faces: Optional[np.ndarray] = None,
    core_mesh_filename: Optional[str] = None,
) -> Dict[str, str]:
    """Write single/dual-material GLBs, Level 2/3 JPEGs, body_scan.json, and optional scan_core.glb."""
    from .scan_service import write_multi_material_glb

    os.makedirs(output_dir, exist_ok=True)
    uv_layout = tex_outputs.uv_layout
    head_uv_layout = getattr(tex_outputs, "head_uv_layout", None)

    all_faces = (
        np.vstack([uv_layout.orig_faces, head_uv_layout.orig_faces])
        if head_uv_layout is not None
        else uv_layout.orig_faces
    )
    full_normals = _compute_smooth_vertex_normals(verts_a, all_faces)

    body_verts = verts_a[uv_layout.vmapping]
    body_faces = uv_layout.uv_faces
    body_normals = full_normals[uv_layout.vmapping]
    body_uvs_gltf = _opengl_uv_to_gltf(uv_layout.uv)

    if head_uv_layout is not None:
        head_verts = verts_a[head_uv_layout.vmapping]
        head_faces = head_uv_layout.uv_faces
        head_normals = full_normals[head_uv_layout.vmapping]
        head_uvs_gltf = _opengl_uv_to_gltf(head_uv_layout.uv)

    def _build_primitives(
        body_tex_bytes: Optional[bytes] = None,
        head_tex_bytes: Optional[bytes] = None,
    ) -> List[Dict[str, Any]]:
        if head_uv_layout is None:
            return [
                {
                    "verts": body_verts,
                    "faces": body_faces,
                    "normals": body_normals,
                    "uvs": body_uvs_gltf,
                    "texture_jpeg_bytes": body_tex_bytes,
                    "base_color_rgba": (0.88, 0.88, 0.88, 1.0),
                    "double_sided": False,
                    "name": "AlphaScanBody",
                }
            ]
        return [
            {
                "verts": body_verts,
                "faces": body_faces,
                "normals": body_normals,
                "uvs": body_uvs_gltf,
                "texture_jpeg_bytes": body_tex_bytes,
                "base_color_rgba": (0.88, 0.88, 0.88, 1.0),
                "double_sided": False,
                "name": "body",
            },
            {
                "verts": head_verts,
                "faces": head_faces,
                "normals": head_normals,
                "uvs": head_uvs_gltf,
                "texture_jpeg_bytes": head_tex_bytes,
                "base_color_rgba": (0.88, 0.88, 0.88, 1.0),
                "double_sided": False,
                "name": "head",
            },
        ]

    glb_path = os.path.join(output_dir, body_json["mesh"]["file"])
    json_path = os.path.join(output_dir, "body_scan.json")

    write_multi_material_glb(_build_primitives(None, None), glb_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(body_json, f, indent=2, ensure_ascii=False)

    outputs: Dict[str, str] = {"glb": glb_path, "json": json_path}

    core_name = core_mesh_filename or body_json.get("mesh", {}).get("core_file")
    if core_name or (core_verts is not None and core_faces is not None):
        target_core_name = core_name or "scan_core.glb"
        cv = np.asarray(verts_a if core_verts is None else core_verts, dtype=np.float32)
        cf = np.asarray(all_faces if core_faces is None else core_faces, dtype=np.int64)
        core_normals = _compute_smooth_vertex_normals(cv, cf)
        core_path = os.path.join(output_dir, target_core_name)
        write_multi_material_glb(
            [
                {
                    "verts": cv,
                    "faces": cf,
                    "normals": core_normals,
                    "base_color_rgba": (0.70, 0.72, 0.75, 1.0),
                    "metallic_factor": 0.06,
                    "roughness_factor": 0.68,
                    "double_sided": False,
                    "name": "AlphaScanCore",
                }
            ],
            core_path,
        )
        outputs["core_glb"] = core_path

    tier_specs: List[tuple] = []
    if privacy_level >= 2:
        tier_specs.append(
            (
                "tex_blur",
                "glb_preview_blur",
                "tex_blur.jpg",
                "scan_preview_blur.glb",
                tex_outputs.tex_blur_rgb,
                getattr(tex_outputs, "head_tex_blur_rgb", None),
            )
        )
    if privacy_level >= 3:
        tier_specs.append(
            (
                "tex_face",
                "glb_preview_face",
                "tex_face.jpg",
                "scan_preview_face.glb",
                tex_outputs.tex_face_rgb,
                getattr(tex_outputs, "head_tex_face_rgb", None),
            )
        )

    for tex_key, glb_key, tex_name, preview_name, body_rgb, head_rgb in tier_specs:
        tex_path = os.path.join(output_dir, tex_name)
        body_bytes = _encode_rgb_jpeg_bytes(body_rgb, quality=94)
        with open(tex_path, "wb") as fp:
            fp.write(body_bytes)
        outputs[tex_key] = tex_path

        if export_preview_glb:
            preview_path = os.path.join(output_dir, preview_name)
            head_bytes = (
                _encode_rgb_jpeg_bytes(head_rgb if head_rgb is not None else body_rgb, quality=94)
                if head_uv_layout is not None
                else None
            )
            write_multi_material_glb(_build_primitives(body_bytes, head_bytes), preview_path)
            outputs[glb_key] = preview_path

    return outputs


def secure_delete_files(
    paths: List[str],
    audit_log_path: str,
    actor: str = "pipeline",
) -> Dict[str, Any]:
    """Delete specified source files from disk and append a minimal audit entry without keeping photo bytes."""
    deleted: List[str] = []
    for path in paths:
        if path and os.path.isfile(path):
            os.remove(path)
            deleted.append(os.path.basename(path))

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "deleted_count": len(deleted),
        "deleted_files": deleted,
    }
    log_dir = os.path.dirname(audit_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def save_debug_plots(
    verts_a: np.ndarray,
    keep_torso_verts: np.ndarray,
    debug_slices: Dict[str, Optional[np.ndarray]],
    output_dir: str,
) -> None:
    """Save A-pose arm mask and 2D torso slice plots for visual inspection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].scatter(verts_a[:, 0], verts_a[:, 1], s=0.3, c=np.where(keep_torso_verts, "#2b7bba", "#e05a47"))
    axes[0].set_title("A-Pose (Blue=Torso, Red=Arm Mask)")
    axes[0].set_aspect("equal")
    axes[1].scatter(verts_a[keep_torso_verts, 0], verts_a[keep_torso_verts, 1], s=0.4, c="#2b7bba")
    axes[1].set_title("Torso Only (Front)")
    axes[1].set_aspect("equal")
    axes[2].scatter(verts_a[keep_torso_verts, 2], verts_a[keep_torso_verts, 1], s=0.4, c="#2b7bba")
    axes[2].set_title("Torso Only (Side, +Z Front)")
    axes[2].set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "debug_apose.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key in zip(axes, ["neck", "chest", "waist", "hip"]):
        pts = debug_slices.get(key)
        if pts is not None and len(pts) >= 3:
            hull = ConvexHull(pts)
            hp = np.vstack([pts[hull.vertices], pts[hull.vertices[0]]])
            ax.scatter(pts[:, 0] * 100, pts[:, 1] * 100, s=4, c="#2b7bba")
            ax.plot(hp[:, 0] * 100, hp[:, 1] * 100, "r-", lw=1.5)
        ax.set_title(key.upper())
        ax.set_aspect("equal")
        ax.set_xlabel("cm")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "debug_slices.png"), dpi=150)
    plt.close(fig)
