"""Escritura de piezas en .obj/.mtl y lectura de .obj con objparser."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import objparser
from .mesh import MATERIALS, Mesh, newell_normal

MTL_NAME = "pieces.mtl"


def write_mtl(path: Path, materials: Dict[str, Tuple[float, float, float]] = MATERIALS):
    lines = ["# Materiales de las piezas de prueba: un color plano por tipo de parte"]
    for name, (r, g, b) in materials.items():
        lines += [f"newmtl {name}", f"Kd {r:.3f} {g:.3f} {b:.3f}", "Ka 1 1 1", "illum 1", ""]
    path.write_text("\n".join(lines))


def write_obj(path: Path, mesh: Mesh, name: str, mtl: str = MTL_NAME):
    """Un objeto, un grupo por material; vértices compartidos y una normal por cara (o, en las
    caras de una superficie curva, una por vértice, con su grupo de suavizado)."""
    vertices: Dict[Tuple[float, ...], int] = {}
    normals: List[Tuple[float, float, float]] = []
    by_material: Dict[str, List[str]] = {}

    def vid(p):
        key = tuple(round(float(c), 6) for c in p)
        if key not in vertices:
            vertices[key] = len(vertices) + 1
        return vertices[key]

    for face in mesh.faces:
        lines = by_material.setdefault(face.material, [])
        if face.normals is not None:
            refs = []
            for p, vn in zip(face.points, face.normals):
                normals.append(tuple(vn))
                refs.append(f"{vid(p)}//{len(normals)}")
            lines += [f"s {face.smooth}", f"f {' '.join(refs)}"]
            continue
        n = newell_normal(face.points)
        n = n / (np.linalg.norm(n) or 1.0)
        normals.append(tuple(n))
        ni = len(normals)
        refs = " ".join(f"{vid(p)}//{ni}" for p in face.points)
        lines += ["s off", f"f {refs}"]

    out = [f"# Pieza {name}: metros, X este, Y arriba, Z sur", f"mtllib {mtl}", f"o {name}"]
    out += [f"v {x:g} {y:g} {z:g}" for (x, y, z) in vertices]
    out += [f"vn {x:.6f} {y:.6f} {z:.6f}" for (x, y, z) in normals]
    for material, faces in by_material.items():
        out += [f"g {material}", f"usemtl {material}"] + faces
    path.write_text("\n".join(out) + "\n")


def read_obj(path: Path) -> List[tuple]:
    """Caras de un .obj como (puntos (n, 3), color difuso, normales por vértice o None, grupo de
    suavizado), usando objparser. Solo las caras de un grupo de suavizado (s 1, s 2…) llevan
    normales por vértice; las demás son planas."""
    objects, materials = objparser.load_model(str(path))
    faces = []
    for obj in objects.values():
        vertices = np.array(obj.v_list, dtype=float)
        normals = np.array(obj.vn_list, dtype=float) if obj.vn_list else None
        for _name, smooth_groups, usemtl in obj.groups:
            mat = materials.get(usemtl)
            color = tuple(float(c) for c in mat.Kd) if mat is not None and mat.Kd else (0.8, 0.8, 0.8)
            for group, face_numbers in smooth_groups.items():
                for n in face_numbers:
                    refs = obj.f_list[n - 1]
                    idx = refs[:, 0] - 1
                    vn = None
                    if group and normals is not None and refs.shape[1] > 2 and (refs[:, 2] > 0).all():
                        vn = normals[refs[:, 2] - 1]
                    faces.append((vertices[idx], color, vn, int(group or 0)))
    return faces
