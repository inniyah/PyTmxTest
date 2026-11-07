"""Escritura de piezas en .obj/.mtl y lectura de .obj con objparser."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import objparser
from .mesh import MATERIALS, Mesh

MTL_NAME = "pieces.mtl"


def write_mtl(path: Path, materials: Dict[str, Tuple[float, float, float]] = MATERIALS):
    lines = ["# Materiales de las piezas de prueba: un color plano por tipo de parte"]
    for name, (r, g, b) in materials.items():
        lines += [f"newmtl {name}", f"Kd {r:.3f} {g:.3f} {b:.3f}", "Ka 1 1 1", "illum 1", ""]
    path.write_text("\n".join(lines))


def write_obj(path: Path, mesh: Mesh, name: str, mtl: str = MTL_NAME):
    """Un objeto, un grupo por material; vértices compartidos y una normal por cara."""
    vertices: Dict[Tuple[float, ...], int] = {}
    normals: List[Tuple[float, float, float]] = []
    by_material: Dict[str, List[str]] = {}

    def vid(p):
        key = tuple(round(float(c), 6) for c in p)
        if key not in vertices:
            vertices[key] = len(vertices) + 1
        return vertices[key]

    for face in mesh.faces:
        n = np.cross(face.points[1] - face.points[0], face.points[2] - face.points[0])
        n = n / (np.linalg.norm(n) or 1.0)
        normals.append(tuple(n))
        ni = len(normals)
        refs = " ".join(f"{vid(p)}//{ni}" for p in face.points)
        by_material.setdefault(face.material, []).append(f"f {refs}")

    out = [f"# Pieza {name}: metros, X este, Y arriba, Z sur", f"mtllib {mtl}", f"o {name}"]
    out += [f"v {x:g} {y:g} {z:g}" for (x, y, z) in vertices]
    out += [f"vn {x:.6f} {y:.6f} {z:.6f}" for (x, y, z) in normals]
    for material, faces in by_material.items():
        out += [f"g {material}", f"usemtl {material}"] + faces
    path.write_text("\n".join(out) + "\n")


def read_obj(path: Path) -> List[Tuple[np.ndarray, Tuple[float, float, float]]]:
    """Caras de un .obj como (puntos (n, 3), color difuso), usando objparser."""
    objects, materials = objparser.load_model(str(path))
    faces = []
    for obj in objects.values():
        vertices = np.array(obj.v_list, dtype=float)
        for _name, smooth_groups, usemtl in obj.groups:
            mat = materials.get(usemtl)
            color = tuple(float(c) for c in mat.Kd) if mat is not None and mat.Kd else (0.8, 0.8, 0.8)
            for face_numbers in smooth_groups.values():
                for n in face_numbers:
                    idx = obj.f_list[n - 1][:, 0] - 1
                    faces.append((vertices[idx], color))
    return faces
