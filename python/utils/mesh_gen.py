import numpy as np
from pathlib import Path
import pymeshlab
import trimesh
import mcubes
import meshio
import pygalmesh
from scipy.interpolate import RegularGridInterpolator

grid_resolution = 200

if __name__ == '__main__':
    
    # Generate watertight triangle mesh from SDF grid
    # root = Path("~/Documents/Physical-stability/main/python/data/chair_1200").expanduser()
    # pattern = "*/sdf/model_normalized.npz"

    # for sdf_path in sorted(root.rglob(pattern)):
    #     sdf_data = np.load(sdf_path)
    #     grid_sdf = sdf_data['grid_sdf']
    #     grid_sdf = grid_sdf.reshape((grid_resolution, grid_resolution, grid_resolution))
    #     vertices, triangles = mcubes.marching_cubes(grid_sdf, 0.0)
    #     import pdb; pdb.set_trace()
    #
    #     watertight_filename = sdf_path.parent / "watertight.obj"
    #     mcubes.export_obj(vertices, triangles, str(watertight_filename))
    #     print(f"generate watertight mesh: {watertight_filename}")
        
    filename = Path("/home/zhiqiang/Documents/Physical-stability/main/python/data/chair_1200/1007e20d5e811b308351982a6e40cf41/sdf/model_normalized.npz")
    sdf_data = np.load(filename)
    
    grid_points = sdf_data['grid_points']
    grid_sdf = sdf_data['grid_sdf']
    grid_sdf = grid_sdf.reshape((grid_resolution, grid_resolution, grid_resolution))

    # vertices, triangles = mcubes.marching_cubes(grid_sdf, 0.0)
    
    # x = np.linspace(-1, 1, grid_resolution)
    # y = np.linspace(-1, 1, grid_resolution)
    # z = np.linspace(-1, 1, grid_resolution)
    
    # # voxel = float(x[1] - x[0])
    # interp = RegularGridInterpolator(
    #     (x, y, z), grid_sdf,
    # )

    # class MyDomain(pygalmesh.DomainBase):
    #     def __init__(self):
    #         super().__init__()
            
    #     def eval(self, x):
    #         if np.any(np.abs(x) > 1.0):
    #             x = np.clip(x, -1.0, 1.0)
    #             p = np.asarray(x, dtype=float)[None, :]
    #             return float(interp(p).item()) 
    #         else:
    #             p = np.asarray(x, dtype=float)[None, :]
    #             return float(interp(p).item())

    #     def get_bounding_sphere_squared_radius(self):
    #         return 3.0
        
    # domain = MyDomain()

    # mesh = pygalmesh.generate_mesh(
    #     domain,
    #     max_cell_circumradius=0.1,      # coarse inside
    # )

    # Rebuild axes (must match how you originally defined the grid!)

    vertices, triangles = mcubes.marching_cubes(grid_sdf, 0.0)
    mcubes.export_obj(vertices, triangles, "test.obj")
    # Post processing in case there are too many triangles
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh("test.obj")
    ms.meshing_isotropic_explicit_remeshing()
    ms.save_current_mesh("test.obj")
    
    tri_mesh = trimesh.load("test.obj")
    triangles = tri_mesh.faces
    vertices = tri_mesh.vertices
    
    cells = [("triangle", triangles.astype(np.int32))]
    meshio.write("test.vtu", meshio.Mesh(vertices, cells))
    mesh = pygalmesh.generate_volume_mesh_from_surface_mesh(
        "test.vtu",
        min_facet_angle=25.0,
        max_radius_surface_delaunay_ball=0.15,
        max_facet_distance=0.008,
        max_circumradius_edge_ratio=3.0,
        verbose=False,
    )

    mesh.write("test.vtk")
    import pyvista as pv
    g = pv.read("test.vtk")
    g.plot(show_edges=True)