from utils.load_matlab import *
from utils.visualize import *
import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import polyscope.imgui as psim
import polyscope.implot as psimplot
import random
from pathlib import Path

# ids = random.sample(range(75), 10)
ids = range(25)
ids_str = []
for i in ids:
    ids_str.append(str(i))
id_selected = ids_str[0]

def callback():
    global id_selected, ids_str
    global meshes, mesh_ps, mesh_ws
    changed = psim.BeginCombo("Pick a random index", id_selected)
    if changed:
        for i in range(10):
            val = ids_str[i]
            _, selected = psim.Selectable(val, id_selected==val)
            if selected:
                id_selected = val
                meshes[i].set_enabled(True)
            else:
                meshes[i].set_enabled(False)
        psim.EndCombo()


if __name__ == "__main__":
    str_name = '~/Documents/Physical-stability/main/python/data/chair_40/'
    shape_name = str('6da85f0549621b837d379e8940a36f92')
    str_name = str_name + shape_name + '/tet/WCSA.mat'
    filename = Path(str_name).expanduser()
    # filename = 'data/test.mat'
    data_dict = load_matlab(filename)

    # results to be visualized
    best_eid = data_dict['results']['best_eid']
    best_wid = data_dict['results']['best_wid']
    p = data_dict['results']['p_all']            # size[75, bV]
    s = data_dict['results']['stress_all']       # size[75, T]
    
    # mesh info: structure and weak regions
    weak_regions = data_dict['weak_regions']
    V = data_dict['parameters']['V']
    NE = data_dict['parameters']['numEigenModes']
    NW = data_dict['parameters']['numWeakRegions']
    N = NW
    bV = data_dict['parameters']['bV']
    T = data_dict['parameters']['T']
    bF = data_dict['parameters']['bF']

    # visualization of the optimized pressure results
    ps.init()

    bVertices = data_dict['mesh_info']['bVertices']
    bFacesVIds = data_dict['mesh_info']['bFacesVIds']
    bFacesTIds = data_dict['mesh_info']['bFacesTIds']
    mapping = data_dict['mesh_info']['mapping']
    bvNorms = data_dict['mesh_info']['bvNorms']
    bvAreas = data_dict['mesh_info']['bvAreas']
    
    vertexPos = data_dict['mesh_info']['vertexPos']
    tetFaces = data_dict['mesh_info']['tetFaces']
    verts = vertexPos.T.copy()
    tets  = tetFaces.T.copy()
    All_bWRIds = weak_regions['All_bWRIds']
    All_WRIds = weak_regions['All_WRIds']
    bvIndices = data_dict['mesh_info']['bvIndices'].squeeze()
    
    # This incopporates all the weak regions of the shape
    # It should have consistency in theory
    weakmap = np.full(V, 0, dtype=float)
    for i in range(N):
        WRIds = All_WRIds[i].squeeze()
        weakmap[WRIds] += 1.0
    threshold = np.quantile(weakmap, 0.999).astype(int)
    weakmap[weakmap > threshold] = threshold
    weakmap = weakmap / weakmap.max()
    weakmap = weakmap ** 0.5

    mesh = ps.register_volume_mesh("mesh", verts, tets, transparency=0.5)
    mesh.add_scalar_quantity('weakmap', weakmap, vminmax=(0., 1.), cmap='viridis', enabled=True)
    ps.show()

    # meshes = []
    # for i in ids:
        # name = f"random_mesh_{i}"
        # mesh = ps.register_surface_mesh(name, bVertices.T, bFacesVIds.T)
        # pressure = p[i][np.newaxis, :] * bvAreas
        # p_max = pressure.max()
        # pressure = (pressure*bvNorms).T
        # pressure = pressure / p_max
        # mesh.add_vector_quantity(f'random_pressure_{i}', pressure)
        
        # bweakregion = weak_regions['All_bWRIds'][i]
        # color = get_bweakregion_color(bweakregion, bV)
        # mesh.add_color_quantity(f'random_weakregion_{i}', color)

        # stress = s[i][np.newaxis, :][0, bFacesTIds].flatten()
        # s_max = stress.max()
        # stress = stress / s_max
        # mesh.add_scalar_quantity(f'random_stress_{i}', stress, defined_on='faces')
        
        # meshes.append(mesh)

    # ps.set_user_callback(callback)
    
    # mesh_o = ps.register_surface_mesh("original_mesh", bVertices.T, bFacesVIds.T)
    # for i in range(15):
    #     if i > 0:
    #         ps.remove_surface_mesh(f"eigenvector_mesh_{i-1}")
    #     eigenvector = eigenvectors[:, :, i]
    #     vertices = bVertices.T + eigenvector[bvIndices, :]
    #     mesh = ps.register_surface_mesh(f"eigenvector_mesh_{i}", vertices, bFacesVIds.T)
    #     ps.show()
