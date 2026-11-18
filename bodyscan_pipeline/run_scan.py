"""CLI entrypoint for running Alpha Scan Step 1 + Step 2 + Step 3 pipeline."""

import argparse
import gc
import glob
import importlib.util
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
try:
    import trimesh
except ImportError:
    trimesh = None  # type: ignore[assignment]

from .config import (
    CARDINAL_VIEWS,
    DEFAULT_RECONSTRUCTION_MODE,
    DEFAULT_TEXTURE_SIZE,
    HEAD_TEXTURE_SIZE,
    MHR_HEAD_SUBDIV_STEPS,
    RECONSTRUCTION_MODES,
)
from .exporter import (
    build_body_scan_json,
    build_texture_metadata,
    export_textured_scan_artifacts,
    save_debug_plots,
    secure_delete_files,
)
from .fitting import evaluate_proportion_audit, fit_mhr_multiview
from .fusion import build_scaled_apose_mesh, fuse_multiview_params
from .head_hair import reconstruct_highres_head_and_hair
from .inference import (
    load_sam3d_estimator,
    run_inference_on_face_image,
    run_inference_on_images,
    save_mhr_head_assets,
    strip_raw_images_for_cache,
)
from .measurements import compute_loft_profile, measure_all_circumferences, validate_sanity
from .outer_shell import build_dual_layer_outer_shell
from .scan_service import (
    _build_fallback_planar_uvs,
    _export_geometry_only_scan_artifacts,
    _measure_circumferences_and_loft_numpy,
    _resolve_effective_mode,
    _transfer_detail_verts_to_view,
)
from .segmentation import process_views_semantics, strip_ram_only_semantics
from .texturing import build_view_observations, extract_mhr_uvs, run_step3_texturing

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def collect_images(folder: str) -> List[str]:
    """Gather sorted image paths from folder."""
    paths: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(set(paths))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for live inference or cached Step 2 execution."""
    parser = argparse.ArgumentParser(description="Alpha Scan Body Scan v2 (SAM 3D Body + MHR)")
    parser.add_argument("--height_cm", type=float, required=True, help="Subject height in cm")
    parser.add_argument("--sex", type=str, choices=["male", "female"], required=True, help="Subject biological sex")
    parser.add_argument("--weight_kg", type=float, default=0.0, help="Subject weight in kg")
    parser.add_argument("--person_id", type=str, default="test_subject_01", help="Subject identifier")
    parser.add_argument("--tier", type=str, default="alpha_scan_plus", choices=["alpha_scan_plus", "alpha_scan_pro"])
    parser.add_argument("--output_dir", type=str, default="./ket_qua_step2", help="Output folder")
    parser.add_argument("--save_debug_plots", action="store_true", help="Save PNG QA plots alongside GLB/JSON")

    # Mode 1: Run full inference from images
    parser.add_argument("--image_folder", type=str, default="", help="Folder containing 4-8 multi-angle photos")
    parser.add_argument("--sam_repo_dir", type=str, default=".", help="Path to cloned facebookresearch/sam-3d-body")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/sam-3d-body-dinov3/model.ckpt")
    parser.add_argument("--mhr_path", type=str, default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")

    # Mode 2: Fast offline re-measurement from cached outputs
    parser.add_argument("--cached_outputs", type=str, default="", help="Optional path to step2_raw_outputs.pt")
    parser.add_argument("--head_assets", type=str, default="./mhr_head_assets.pt", help="Path to mhr_head_assets.pt")

    # Mode 3: Step 3 Texture Baking & Privacy Tiers
    parser.add_argument(
        "--enable_texture",
        action="store_true",
        help="Enable Step 3 UV unwrap & texture baking (required to export tex_blur/tex_face when --privacy_level >= 2)",
    )
    parser.add_argument("--face_image", type=str, default="", help="Optional 1m closeup front face photo")
    parser.add_argument("--texture_size", type=int, default=DEFAULT_TEXTURE_SIZE, help="Output texture width/height")
    parser.add_argument("--privacy_level", type=int, default=3, choices=[1, 2, 3], help="1=plain, 2=blur, 3=clear")
    parser.add_argument(
        "--reconstruction_mode",
        type=str,
        default=DEFAULT_RECONSTRUCTION_MODE,
        choices=list(RECONSTRUCTION_MODES),
        help="3D reconstruction mode: calibrated_mhr, hi_res_head, or dual_layer",
    )
    parser.add_argument("--no_preview_glb", action="store_true", help="Skip exporting textured preview GLB files")
    parser.add_argument(
        "--delete_inputs",
        action="store_true",
        help="Explicitly delete input image files after processing and write deletion_audit.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    """Execute multi-view fusion, fitting, high-res head/outer shell, measurements, texturing, and export."""
    args = parse_args()
    closeup_face_result: Optional[dict] = None
    image_paths: List[str] = []
    raw_results: List[dict] = []
    segmented_views: List[Any] = []
    warnings: List[str] = []
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.cached_outputs:
            raw_results = torch.load(args.cached_outputs, weights_only=False)
            head_assets = torch.load(args.head_assets, map_location="cpu", weights_only=True)
            mhr_model = torch.jit.load(args.mhr_path, map_location="cpu")
        else:
            if not args.image_folder:
                raise ValueError("Provide either --image_folder or --cached_outputs.")
            image_paths = collect_images(args.image_folder)
            if not image_paths:
                raise FileNotFoundError(f"No images found in {args.image_folder}")

            want_high_detail = args.reconstruction_mode in ("hi_res_head", "dual_layer") and len(image_paths) >= 4
            want_depth = want_high_detail
            model, estimator = load_sam3d_estimator(args.sam_repo_dir, args.checkpoint_path, args.mhr_path)
            save_mhr_head_assets(model, args.head_assets)
            raw_results = run_inference_on_images(
                estimator,
                image_paths,
                keep_rgb=True,
                keep_head_crop=want_high_detail,
                head_max_side=3200,
                extract_depth=want_depth,
            )
            torch.save(
                strip_raw_images_for_cache(raw_results),
                os.path.join(args.output_dir, "step2_raw_outputs.pt"),
            )
            if args.enable_texture and args.privacy_level == 3 and args.face_image:
                closeup_face_result = run_inference_on_face_image(estimator, args.face_image, keep_rgb=True)
            if raw_results:
                segmented_views = process_views_semantics(raw_results, estimator=estimator)
            head = model.head_pose
            head_assets = {
                "scale_mean": head.scale_mean.detach(),
                "scale_comps": head.scale_comps.detach(),
                "faces": head.faces.detach().cpu(),
            }
            mhr_model = head.mhr
            del estimator, model, head
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cardinal_labels = {name for name, _ in CARDINAL_VIEWS}
        unique_cardinals = (
            len({sv.view_label for sv in segmented_views if sv.view_label in cardinal_labels})
            if segmented_views
            else 0
        )
        effective_mode, mode_warnings = _resolve_effective_mode(args.reconstruction_mode, unique_cardinals)
        for w in mode_warnings:
            if w not in warnings:
                warnings.append(w)
        for sv in segmented_views:
            for w in sv.warnings:
                if w not in warnings:
                    warnings.append(w)

        fused = fuse_multiview_params(raw_results)
        proportion_audit = None
        fit_debug: Dict[str, Any] = {}
        if segmented_views:
            fused, proportion_audit, fit_debug = fit_mhr_multiview(
                mhr_model=mhr_model,
                head_assets=head_assets,
                fused=fused,
                segmented_views=segmented_views,
                raw_results=raw_results,
                height_cm=float(args.height_cm),
                weight_kg=float(args.weight_kg or 0.0),
                sex=args.sex,
                steps=50,
                prefer_pytorch3d=True,
            )
            fit_debug = fit_debug if isinstance(fit_debug, dict) else {}

        cached_apose_params = fit_debug.get("apose_params")
        apose_kwargs: Dict[str, Any] = {}
        if cached_apose_params is not None:
            apose_kwargs["apose_params"] = cached_apose_params
        verts_a, joints_a, keep_torso_verts, raw_height_cm = build_scaled_apose_mesh(
            mhr_model, head_assets, fused, args.height_cm, **apose_kwargs
        )

        faces_raw = head_assets["faces"]
        faces = (
            faces_raw.detach().cpu().numpy()
            if hasattr(faces_raw, "detach")
            else np.asarray(faces_raw, dtype=np.int64)
        )
        if proportion_audit is None:
            proportion_audit = evaluate_proportion_audit(
                verts_a=verts_a,
                faces=faces,
                sex=args.sex,
                declared_weight_kg=float(args.weight_kg or 0.0),
                mhr_silhouette_iou=0.0,
            )
        for w in proportion_audit.warnings:
            if w not in warnings:
                warnings.append(w)

        height_m = args.height_cm / 100.0
        if trimesh is not None:
            full_mesh = trimesh.Trimesh(vertices=verts_a, faces=faces, process=False)
            torso_face_mask = keep_torso_verts[faces].all(axis=1)
            torso_mesh = trimesh.Trimesh(vertices=verts_a, faces=faces[torso_face_mask], process=False)
            circumferences, debug_slices = measure_all_circumferences(
                full_mesh, torso_mesh, joints_a, height_m, args.sex
            )
            loft = compute_loft_profile(torso_mesh, height_m)
        else:
            circumferences, debug_slices, loft = _measure_circumferences_and_loft_numpy(
                verts_a, faces, keep_torso_verts, joints_a, height_m, args.sex
            )
        for w in validate_sanity(circumferences, args.sex):
            if w not in warnings:
                warnings.append(w)

        export_verts = verts_a
        export_faces = faces
        head_vert_mask: Optional[np.ndarray] = None
        hair_vert_mask: Optional[np.ndarray] = None
        face_skin_vert_mask: Optional[np.ndarray] = None
        core_verts: Optional[np.ndarray] = None
        core_faces: Optional[np.ndarray] = None
        core_mesh_filename: Optional[str] = None

        if segmented_views and effective_mode in ("hi_res_head", "dual_layer"):
            head_out = reconstruct_highres_head_and_hair(
                verts_a=verts_a,
                faces=faces,
                joints_a=joints_a,
                segmented_views=segmented_views,
                raw_results=raw_results,
                mhr_model=mhr_model,
                subdiv_steps=MHR_HEAD_SUBDIV_STEPS,
            )
            for w in head_out.warnings:
                if w not in warnings:
                    warnings.append(w)
            export_verts = head_out.full_verts
            export_faces = head_out.full_faces
            export_undisplaced_verts = getattr(head_out, "undisplaced_verts", None)
            head_vert_mask = head_out.head_vert_mask
            hair_vert_mask = head_out.hair_vert_mask
            face_skin_vert_mask = head_out.face_skin_vert_mask

            if effective_mode == "dual_layer":
                outer_out = build_dual_layer_outer_shell(
                    inner_verts_a=verts_a,
                    inner_faces=faces,
                    joints_a=joints_a,
                    highres_head=head_out,
                    segmented_views=segmented_views,
                    raw_results=raw_results,
                    fitted_cameras=fit_debug.get("final_cameras"),
                    fitted_view_verts=fit_debug.get("view_posed_verts"),
                    prefer_pytorch3d=True,
                )
                for w in outer_out.warnings:
                    if w not in warnings:
                        warnings.append(w)
                proportion_audit.outer_silhouette_iou = round(float(outer_out.outer_silhouette_iou), 4)
                export_verts = outer_out.outer_verts
                export_faces = outer_out.outer_faces
                export_undisplaced_verts = getattr(outer_out, "undisplaced_verts", export_undisplaced_verts)
                head_vert_mask = outer_out.head_vert_mask
                hair_vert_mask = outer_out.hair_vert_mask
                face_skin_vert_mask = outer_out.face_skin_vert_mask
                core_verts = outer_out.inner_mhr_verts
                core_faces = outer_out.inner_mhr_faces
                core_mesh_filename = "scan_core.glb"

        audit_dict = proportion_audit.to_dict()

        textures_meta = None
        tex_outputs = None
        if args.enable_texture and args.privacy_level >= 2:
            final_cameras = fit_debug.get("final_cameras")
            view_posed_verts = fit_debug.get("view_posed_verts")
            obs_kwargs: Dict[str, Any] = {
                "weights": fused.weights,
                "closeup_face_result": closeup_face_result,
                "segmented_views": segmented_views or None,
            }
            if final_cameras:
                obs_kwargs["fitted_cameras"] = final_cameras
                if view_posed_verts:
                    obs_kwargs["fitted_view_verts"] = view_posed_verts
            observations = build_view_observations(raw_results, **obs_kwargs)
            if observations:
                has_detail_diff = (len(export_verts) != len(verts_a)) or (
                    not np.allclose(export_verts, verts_a, atol=1e-6)
                )
                if has_detail_diff:
                    for obs in observations:
                        if len(obs.pred_vertices) == len(verts_a):
                            obs.pred_vertices = _transfer_detail_verts_to_view(
                                detail_verts_a=export_verts,
                                base_verts_a=verts_a,
                                view_verts_b=obs.pred_vertices,
                                base_faces=faces,
                                undisplaced_verts_a=export_undisplaced_verts,
                            )
                uvs_arg, uv_faces_arg = extract_mhr_uvs(mhr_model, head_assets, len(export_faces))
                uvs_usable = (
                    uvs_arg is not None
                    and (uv_faces_arg is not None or len(uvs_arg) == len(export_verts))
                )
                if not uvs_usable and importlib.util.find_spec("xatlas") is None:
                    uvs_arg, uv_faces_arg = _build_fallback_planar_uvs(export_verts, export_faces)
                elif not uvs_usable:
                    uvs_arg, uv_faces_arg = None, None

                tex_outputs = run_step3_texturing(
                    verts_a=export_verts,
                    faces=export_faces,
                    joints_a=joints_a,
                    observations=observations,
                    mhr_model=mhr_model,
                    head_assets=head_assets,
                    tex_size=args.texture_size,
                    mhr_uvs=uvs_arg,
                    mhr_uv_faces=uv_faces_arg,
                    head_vert_mask=head_vert_mask,
                    hair_vert_mask=hair_vert_mask,
                    face_skin_vert_mask=face_skin_vert_mask,
                    dual_material=(head_vert_mask is not None),
                    head_tex_size=HEAD_TEXTURE_SIZE,
                )
                textures_meta = build_texture_metadata(
                    tex_outputs,
                    privacy_level=args.privacy_level,
                    export_preview_glb=not args.no_preview_glb,
                    views_used=len(raw_results),
                )
            else:
                warnings.append("Texture requested, but no camera/image views were available (check pred_cam_t/image paths).")
        elif args.privacy_level >= 2 and not args.enable_texture:
            msg = (
                f"Texturing skipped (--privacy_level={args.privacy_level}): "
                "pass --enable_texture to bake tex_blur/tex_face textures and preview GLBs; "
                "exporting plain geometry scan.glb only."
            )
            warnings.append(msg)
            print(f"[info] {msg}")
        elif args.enable_texture and args.privacy_level < 2:
            warnings.append(
                "--enable_texture has no effect with --privacy_level=1 (Level 1 exports plain geometry only)."
            )

        body_json = build_body_scan_json(
            person_id=args.person_id,
            tier=args.tier,
            sex=args.sex,
            height_cm=args.height_cm,
            weight_kg=args.weight_kg,
            circumferences_cm=circumferences,
            loft=loft,
            fused=fused,
            textures=textures_meta,
            reconstruction_mode=args.reconstruction_mode,
            effective_reconstruction_mode=effective_mode,
            proportion_audit=audit_dict,
            core_mesh_filename=core_mesh_filename,
            warnings=warnings,
        )
        if tex_outputs is not None:
            outputs = export_textured_scan_artifacts(
                verts_a=export_verts,
                body_json=body_json,
                output_dir=args.output_dir,
                tex_outputs=tex_outputs,
                export_preview_glb=not args.no_preview_glb,
                privacy_level=args.privacy_level,
                core_verts=core_verts,
                core_faces=core_faces,
                core_mesh_filename=core_mesh_filename,
            )
        else:
            outputs = _export_geometry_only_scan_artifacts(
                verts_a=export_verts,
                faces=export_faces,
                body_json=body_json,
                output_dir=args.output_dir,
                head_vert_mask=head_vert_mask,
                core_verts=core_verts,
                core_faces=core_faces,
                core_mesh_filename=core_mesh_filename,
            )

        if args.save_debug_plots:
            save_debug_plots(verts_a, keep_torso_verts, debug_slices, args.output_dir)

        if args.delete_inputs and (image_paths or args.face_image):
            del_targets = list(image_paths) + ([args.face_image] if args.face_image else [])
            audit = secure_delete_files(
                del_targets,
                audit_log_path=os.path.join(args.output_dir, "deletion_audit.jsonl"),
                actor="cli_run_scan",
            )
            outputs["deletion_audit"] = os.path.join(args.output_dir, "deletion_audit.jsonl")
            print(f"Deleted {audit['deleted_count']} input photos & recorded audit log.")

        print(f"Raw MHR height: {raw_height_cm:.1f} cm -> Scaled: {args.height_cm:.1f} cm")
        print(f"Mode: {effective_mode} (requested: {args.reconstruction_mode}) | Audit: {audit_dict.get('status')}")
        print(f"Images used: {fused.images_used}/{fused.images_total}")
        for k, v in circumferences.items():
            print(f"  {k:12s}: {v:5.1f} cm")
        if warnings:
            print("Warnings:", warnings)
        print("Exported:", outputs)
    finally:
        if segmented_views:
            for sv in segmented_views:
                sv.face_landmarks_ram = None
                sv.nose_profile_curve = None
            segmented_views.clear()
        if raw_results is not None:
            orig_raw = raw_results
            raw_results = strip_ram_only_semantics(orig_raw)
            for item in orig_raw:
                if isinstance(item, dict):
                    item.clear()
            orig_raw.clear()
            if raw_results is not orig_raw:
                for item in raw_results:
                    if isinstance(item, dict):
                        item.clear()
                raw_results.clear()
        if closeup_face_result is not None:
            closeup_face_result.clear()
        gc.collect()


if __name__ == "__main__":
    main()
