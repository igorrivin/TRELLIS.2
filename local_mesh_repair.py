import os
from dataclasses import dataclass

import trimesh


@dataclass
class LocalMeshRepairResult:
    input_path: str
    output_path: str
    watertight_before: bool
    watertight_after: bool
    mesh_count: int
    details: str

    @property
    def repaired(self) -> bool:
        return self.watertight_after and self.output_path != self.input_path


def repair_mesh_locally(input_path: str, output_dir: str, *, output_name: str = "local_repaired.glb") -> LocalMeshRepairResult:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    try:
        scene = trimesh.load(input_path, force="scene")
    except Exception as exc:
        return LocalMeshRepairResult(
            input_path=input_path,
            output_path=input_path,
            watertight_before=False,
            watertight_after=False,
            mesh_count=0,
            details=f"Could not load mesh locally: {exc}",
        )

    meshes = [geometry for geometry in scene.dump(concatenate=False) if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0]
    if not meshes:
        return LocalMeshRepairResult(
            input_path=input_path,
            output_path=input_path,
            watertight_before=False,
            watertight_after=False,
            mesh_count=0,
            details="No mesh geometry found.",
        )

    watertight_before = all(mesh.is_watertight for mesh in meshes)
    if watertight_before:
        return LocalMeshRepairResult(
            input_path=input_path,
            output_path=input_path,
            watertight_before=True,
            watertight_after=True,
            mesh_count=len(meshes),
            details="Already watertight.",
        )

    repaired_meshes = [repair_single_mesh(mesh) for mesh in meshes]
    watertight_after = all(mesh.is_watertight for mesh in repaired_meshes)
    if not watertight_after:
        failed = sum(1 for mesh in repaired_meshes if not mesh.is_watertight)
        return LocalMeshRepairResult(
            input_path=input_path,
            output_path=input_path,
            watertight_before=watertight_before,
            watertight_after=False,
            mesh_count=len(meshes),
            details=f"Local repair left {failed} of {len(meshes)} mesh parts non-watertight.",
        )

    trimesh.Scene(repaired_meshes).export(output_path)
    return LocalMeshRepairResult(
        input_path=input_path,
        output_path=output_path,
        watertight_before=watertight_before,
        watertight_after=True,
        mesh_count=len(meshes),
        details="Local repair produced a watertight mesh.",
    )


def repair_single_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    repaired = mesh.copy()
    try:
        repaired.update_faces(repaired.nondegenerate_faces())
        repaired.update_faces(repaired.unique_faces())
    except Exception:
        pass

    try:
        repaired.merge_vertices()
        repaired.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        trimesh.repair.fix_winding(repaired)
        trimesh.repair.fix_normals(repaired, multibody=True)
        trimesh.repair.fill_holes(repaired)
        trimesh.repair.fix_inversion(repaired, multibody=True)
    except Exception:
        pass

    try:
        repaired.remove_unreferenced_vertices()
    except Exception:
        pass
    return repaired
