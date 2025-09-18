# this utility file contain functions
# load and clean .obj files
# generate tetrahedron mesh from surface mesh
# saving the mesh into .m file with certain format
import numpy as np
from pathlib import Path
import trimesh
import gmsh
import meshio
import pymeshlab as ml

def load_obj(data_dir):
    # All returned values are string/path
    ids = []
    objs = []
    for path in data_dir.rglob('*.obj'):
        shape_id = path.parent.parent.name
        ids.append(shape_id)
        objs.append(path)
    return ids, objs

def clean_obj(objs, watertight_required=False):
    cleaned_meshes = []
    for obj in objs:
        mesh = trimesh.load_mesh(obj, force='mesh', process=False)
        
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        
        mesh = mesh.fill_holes()
        mesh.fix_normals()
        
        if watertight_required and not mesh.is_watertight:
            raise ValueError(f"Mesh {obj} is not watertight after cleaning.")

        cleaned_meshes.append(mesh)
    return cleaned_meshes



