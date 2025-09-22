import os
import numpy as np
from pathlib import Path
import pymeshlab
import trimesh
import mcubes
import meshio
import mesh2sdf
import subprocess
import polyscope as ps
import pickle
import time
import threading
import sys
import argparse
import signal
import traceback

grid_resolution = 200

def get_watertight_mesh(model_filename):
    mesh = trimesh.load(model_filename, skip_materials=True, process=True, force='mesh')
    
    tet_dir = model_filename.parent.parent / "tet"
    os.makedirs(tet_dir, exist_ok=True)
    
    mesh_scale = 0.8
    size = 128
    level = 2 / size
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)

    sdf_filename = tet_dir / "model_sdf.npz"
    np.savez(sdf_filename, sdf=sdf)

    watertight_filename = tet_dir / "model_watertight.obj"
    mesh.vertices = mesh.vertices / scale + center
    mesh.export(str(watertight_filename))

    return watertight_filename

def write_smesh(path, vertices, triangles,
                point_attributes=None, point_markers=None,
                facet_markers=None,
                holes=None,
                regions=None):
    """
    Save a TetGen .smesh file.

    Parameters
    ----------
    path : str or Path
        Output filename (e.g., "mesh.smesh").
    vertices : (N, 3) array_like
        XYZ coordinates.
    triangles : (M, 3) array_like (int)
        Triangle vertex indices (0-based or 1-based; function auto-fixes to 1-based).
    point_attributes : (N, A) array_like or None
        Optional per-point attributes to write after xyz. A can be 1..K. Default: none.
    point_markers : (N,) array_like of int or None
        Optional per-point boundary marker. If provided, boundary marker flag becomes 1.
    facet_markers : (M,) array_like of int or None
        Optional per-facet boundary marker. If provided, boundary marker flag in Part 2 is 1.
    holes : list[(x,y,z)] or None
        Optional list of hole points.
    regions : list[(x,y,z, region_number, region_attribute)] or None
        Optional list of region specs.

    Notes
    -----
    - Part 2 uses the simple per-facet line format:
      "<#corners> v1 v2 ... [boundary_marker]"
      (one triangle per line).
    - Indices are written 1-based, as TetGen expects.
    """
    path = Path(path)
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(triangles, dtype=int)

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if F.ndim != 2 or F.shape[1] < 3:
        raise ValueError("triangles must have shape (M, 3)")

    # Ensure 1-based indexing for faces
    if F.min() == 0:
        F = F + 1

    # Point attributes
    if point_attributes is None:
        PATTR = None
        num_attr = 0
    else:
        PATTR = np.asarray(point_attributes)
        if PATTR.shape[0] != V.shape[0]:
            raise ValueError("point_attributes must have same length as vertices")
        num_attr = PATTR.shape[1] if PATTR.ndim == 2 else 1
        if PATTR.ndim == 1:
            PATTR = PATTR[:, None]

    # Point markers
    if point_markers is None:
        point_marker_flag = 0
    else:
        PM = np.asarray(point_markers, dtype=int)
        if PM.shape[0] != V.shape[0]:
            raise ValueError("point_markers must have same length as vertices")
        point_marker_flag = 1

    # Facet markers
    if facet_markers is None:
        facet_marker_flag = 0
    else:
        FM = np.asarray(facet_markers, dtype=int)
        if FM.shape[0] != F.shape[0]:
            raise ValueError("facet_markers must have same length as triangles")
        facet_marker_flag = 1

    # Holes
    holes = [] if holes is None else list(holes)

    # Regions
    regions = [] if regions is None else list(regions)

    with path.open("w") as f:
        # ---- Part 1: Node list
        f.write(f"{V.shape[0]} 3 {num_attr} {point_marker_flag}\n")
        for i, (x, y, z) in enumerate(V, start=1):
            line = [str(i), f"{x:.17g}", f"{y:.17g}", f"{z:.17g}"]
            if num_attr:
                line += [f"{a:.17g}" for a in PATTR[i-1]]
            if point_marker_flag:
                line.append(str(int(point_markers[i-1])))
            f.write(" ".join(line) + "\n")

        # ---- Part 2: Facet list (triangles as single polygons, no per-facet holes)
        f.write(f"{F.shape[0]} {facet_marker_flag}\n")
        for idx in range(F.shape[0]):
            v = F[idx]
            line = ["3", str(int(v[0])), str(int(v[1])), str(int(v[2]))]
            if facet_marker_flag:
                line.append(str(int(facet_markers[idx])))
            f.write(" ".join(line) + "\n")

        # ---- Part 3: Hole list
        f.write(f"{len(holes)}\n")
        for i, (hx, hy, hz) in enumerate(holes, start=1):
            f.write(f"{i} {hx:.17g} {hy:.17g} {hz:.17g}\n")

        # ---- Part 4: Region attributes list
        f.write(f"{len(regions)}\n")
        for i, reg in enumerate(regions, start=1):
            if len(reg) != 5:
                raise ValueError("Each region must be (x, y, z, region_number, region_attribute)")
            rx, ry, rz, rnum, rattr = reg
            f.write(f"{i} {rx:.17g} {ry:.17g} {rz:.17g}{int(rnum)}{float(rattr):.17g}\n")
            # If you prefer spaces between rnum and rattr (often clearer), use:
            # f.write(f"{i} {rx:.17g} {ry:.17g} {rz:.17g} {int(rnum)} {float(rattr):.17g}\n")

def read_tetgen(path):
    # path needs to be a .1.ele or .1.node file
    mesh = meshio.read(path)
    vertices = mesh.points
    tets = mesh.cells_dict.get('tetra', None)
    return vertices, tets

def write_MATLAB_mesh(path, vertices, tets):
    # generate a .m file at the given path
    vertices = np.asarray(vertices)
    tets = np.asarray(tets)
    V = vertices.shape[0]

    with open(path, "w") as f:
        f.write("clear msh;\n")
        f.write(f"msh.nbNod = {V};\n")
        
        # POS
        f.write("msh.POS = [\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]};\n")
        f.write("];\n")
        
        f.write("msh.MAX = max(msh.POS);\n")
        f.write("msh.MIN = min(msh.POS);\n")

        # TETS
        f.write("msh.TETS = [\n")
        for tet in tets:
            v1, v2, v3, v4 = tet + 1
            f.write(f" {v1} {v2} {v3} {v4} 0\n")
        f.write("];\n")
    print(f"{path} generated")

def run_with_timeout(cmd, timeout):
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate(timeout=timeout)
        returncode = proc.returncode
        return returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise TimeoutError(f"Command {' '.join(cmd)} timed out after {timeout}s")

def process_shape(model_filename, root, success_shape):
    relative_path = model_filename.relative_to(root)
    shape_id = relative_path.parts[0]
    
    # skip if already done
    tetmesh_filename = model_filename.parent.parent / "tet" / "tetmesh.m"
    if tetmesh_filename.exists():
        print(f"Skipping existing: {shape_id}")
        success_shape.append(shape_id)
        return 0
    
    watertight_filename = get_watertight_mesh(model_filename)
    print(f"Watertight mesh generated: {shape_id}")

    meshlab_filename = watertight_filename.parent / "model_simplified.obj"
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(watertight_filename))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum = 5000,
        qualitythr = 1.0,
        preserveboundary = True,
        preservenormal = True,
        preservetopology = True,
        planarquadric = True,
        planarweight = 5e-5,
        )
    ms.compute_selection_by_self_intersections_per_face()
    ms.meshing_remove_selected_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_connected_component_by_face_number()
    ms.meshing_close_holes()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_close_holes()
    ms.meshing_re_orient_faces_coherently()
    ms.save_current_mesh(str(meshlab_filename))
    

    smesh_filename = watertight_filename.parent / "surface.smesh"
    tri_mesh = trimesh.load(meshlab_filename)
    triangles = tri_mesh.faces
    vertices = tri_mesh.vertices
    write_smesh(smesh_filename, vertices, triangles)
    cmd = ["tetgen", "-pqa5e-5R", str(smesh_filename)]
    timeout = 60
    try:
        returncode, stdout, stderr = run_with_timeout(cmd, timeout)
        if returncode != 0:
            raise RuntimeError(f"TetGen failed: {stderr}")
    except TimeoutError:
        print(f"[TIMEOUT] {shape_id} took longer than 60s, skipping...")
        return  # just skip to next shape
    except RuntimeError as e:
        print(f"[FAIL] {shape_id}: {e}")
        return
    
    print(f"TetGen finished for {shape_id}")

    ele_filename = smesh_filename.parent / "surface.1.ele"
    MATLAB_filename = smesh_filename.parent / "tetmesh.m"
    vertices, tets = read_tetgen(ele_filename)
    write_MATLAB_mesh(MATLAB_filename, vertices, tets)
    print(f"MATLAB mesh file generated: {shape_id}")
    
    success_shape.append(shape_id)

def run_worker(shape_path, root):
    success_shape = []
    try:
        print(f"Processing single shape: {shape_path}")
        process_shape(shape_path, root, success_shape)
        sys.exit(0)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, help="Process a single model file and exit")
    parser.add_argument("--root", type=str, default="~/Documents/Physical-stability/main/python/data/chair_1200")
    parser.add_argument("--pattern", type=str, default="*/models/model_normalized.obj")
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    
    if args.shape:
        shape = Path(args.shape)
        run_worker(shape, root)
    
    pattern = args.pattern
    success_shape = []
    shapelist_filename = root / "shape_ids.txt"

    # Optional: append logs per-shape
    log_dir = root / "_logs"
    log_dir.mkdir(exist_ok=True)

    for model_filename in sorted(root.rglob(pattern)):
        relative_path = model_filename.relative_to(root)
        shape_id = relative_path.parts[0]

        # Launch a fresh interpreter for each shape
        cmd = [sys.executable, "-m", "utils.mesh_gen", "--shape", str(model_filename)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            # Parse stdout to see if we actually processed (optional),
            # or simply check tetmesh presence and add on success:
            tetmesh_path = model_filename.parent.parent / "tet" / "tetmesh.m"
            if tetmesh_path.exists():
                success_shape.append(shape_id)
            print(f"[DONE] {shape_id}")
        else:
            # If negative, process died by signal (e.g., SIGABRT = 6)
            if proc.returncode < 0:
                try:
                    sig_name = signal.Signals(-proc.returncode).name
                except ValueError:
                    sig_name = f"SIG{-proc.returncode}"
                print(f"[CRASH] {shape_id} terminated by {sig_name}")
            else:
                print(f"[FAIL] {shape_id} exited with code {proc.returncode}")

            # Save stderr for later debugging, but keep going
            with open(log_dir / f"{shape_id}.stderr.txt", "w") as f:
                f.write(proc.stderr or "")

            continue

    # Write the (newly) successful shape_ids
    with open(shapelist_filename, "w") as f:
        for sid in success_shape:
            f.write(f"{sid}\n")

    print(f"[SUMMARY] {len(success_shape)} shapes succeeded.")