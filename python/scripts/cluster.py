# Using diffustion map and HDBSCAN clustering method to cluster weak regions
# Or using a graph-based method to find the connected components of the graph
from utils.load_matlab import *
from utils.visualize import *
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian, connected_components
from sklearn.cluster import HDBSCAN
import polyscope as ps

def Laplacian_Matrix(vertexPos, tetFaces):

    def dihedral_angle(tet_pos, k, l, i, j):
        """
        Dihedral angle (radians) along edge (i,j) between faces (i,j,k) and (i,j,l).
        Uses the robust atan2 formula with edge-orthogonal face normals.
        """
        v_i = tet_pos[i]
        v_j = tet_pos[j]
        v_k = tet_pos[k]
        v_l = tet_pos[l]

        ki = np.linalg.norm(v_i - v_k)
        kj = np.linalg.norm(v_j - v_k)
        kl = np.linalg.norm(v_l - v_k)
        ij = np.linalg.norm(v_j - v_i)
        il = np.linalg.norm(v_l - v_i)
        jl = np.linalg.norm(v_l - v_j)

        c_ikj = (ki**2+kj**2-ij**2)/(2*ki*kj)
        c_jkl = (kj**2+kl**2-jl**2)/(2*kj*kl)
        c_ikl = (ki**2+kl**2-il**2)/(2*ki*kl)
        s_jkl = np.sin(np.arccos(c_jkl))
        s_ikl = np.sin(np.arccos(c_ikl))

        c_ang = (c_ikj-c_jkl*c_ikl)/(s_jkl*s_ikl)
        ang = np.arccos(c_ang)
        return ang

    V = vertexPos.shape[1]
    T = tetFaces.shape[1]
    LM_rows = []
    LM_cols = []
    LM_vals = []

    for i in range(T):
        tet_idx = tetFaces[:, i]
        tet_pos = vertexPos[:, tet_idx].T  # size[4, 3]
        edges = {
            (0,1):(2,3),
            (0,2):(1,3),
            (0,3):(1,2),
            (1,2):(0,3),
            (1,3):(0,2),
            (2,3):(0,1),
        }
        for (i, j), (k, l) in edges.items():
            ang = dihedral_angle(tet_pos, k, l, i, j)
            length = np.linalg.norm(tet_pos[k] - tet_pos[l])
            LM_rows.append(tet_idx[i])
            LM_cols.append(tet_idx[j])
            val = (1/6)*length*(1/(np.tan(ang) + 1e-8))
            LM_vals.append(val)
    LM = sp.coo_matrix((LM_vals, (LM_rows, LM_cols)), shape=(V, V))
    LM.sum_duplicates()
    LM = LM + LM.transpose()
    A = LM
    LM, D = laplacian(LM, return_diag=True)
    D = sp.diags(D)
    return LM, D, A

def Eigen_Space(LM, eigens=100):
    # At last we just compute the eigenvectors of matrix LM
    evals, evecs = eigsh(LM, k=eigens+1, which='SM')
    order = np.argsort(evals)              # ascending
    evals_sorted = evals[order]
    evecs_sorted = evecs[:, order]
    # We will eliminate the first eigenvector in later projection part
    return np.real(evals_sorted), np.real(evecs_sorted)

# ps.init()

# vol = ps.register_volume_mesh("tet mesh", verts, tets=tets, interior_color=(0.9,0.9,0.9))
# vol.set_transparency(0.2)

# # Add selected eigenmodes as per-vertex scalar quantities
# # for idx in range(60):
# #     name = f"phi_{idx+1} (lambda={evals[idx]:.3g})"
# #     vol.add_scalar_quantity(name, evecs[:, idx],
# #                             defined_on='vertices', cmap='coolwarm', enabled=True)

# # simple test to see if the projection if working
# WRIds1 = OPT['weakRegions'][0, 0][0, 11][0, 2] - 1
# indicator1 = np.zeros((V, 1))
# indicator1[WRIds1] = 1.0

# WRIds2 = OPT['weakRegions'][0, 0][0, 1][0, 3] - 1
# indicator2 = np.zeros((V, 1))
# indicator2[WRIds2] = 1.0

def project_one_indicator(evecs, evals, indicator):
    p_coord = np.linalg.inv(evecs.T@evecs)@evecs.T@indicator
    # normalize p
    # normalize with evals to raise the importance of low-freq components
    p_coord = p_coord.flatten()[1:]
    evals = evals[1:]
    evecs = evecs[:, 1:]
    p_coord = p_coord / (evals**0.5)
    # normalize to make sure the norm is 1
    p_coord_norm = np.linalg.norm(p_coord)
    p_coord = p_coord / p_coord_norm
    p_indicator = evecs@p_coord
    return p_coord, p_indicator

def project_indicators(evecs, evals, indicators):
    # prjection-based method
    # project N indicators into eigenspace and normalize it
    p_coords = np.linalg.inv(evecs.T@evecs)@evecs.T@indicators
    p_coords = p_coords[1:, :]
    # First, normalize to make small eigenvalues more important
    p_coords = p_coords / (evals[1:]**0.5).reshape(-1, 1)
    # Second, normalize to make the projected coord unit
    p_coords_norm = np.linalg.norm(p_coords, axis=0)
    p_coords = p_coords / (p_coords_norm.reshape(1, -1))

    return p_coords

# p_coord_1 = project_indicator(evecs, evals, indicator1)
# p_coord_2 = project_indicator(evecs, evals, indicator2)
# evecs = evecs[:, 1:]
# evals = evals[1:]
# p_indicator_1 = evecs @ p_coord_1
# p_indicator_2 = evecs @ p_coord_2

# vol.add_scalar_quantity('wr1', indicator1.flatten(), defined_on='vertices', cmap='viridis', enabled=True)
# vol.add_scalar_quantity('wrp1', p_indicator_1.flatten(), defined_on='vertices', cmap='viridis', enabled=True)
# vol.add_scalar_quantity('wr2', indicator2.flatten(), defined_on='vertices', cmap='viridis', enabled=True)
# vol.add_scalar_quantity('wrp2', p_indicator_2.flatten(), defined_on='vertices', cmap='viridis', enabled=True)
# ps.show()

def HDBSCAN_Clustering(Indicators,
                       vertexPos, tetFaces,
                       min_cluster_size=2,
                       min_samples=2,
                       cluster_selection_epsilon=0.6,
                       cluster_selection_method='eom',
                       ):
    # Project indicators into eigenspace with normalization
    LM, D, A = Laplacian_Matrix(vertexPos, tetFaces)
    evals, evecs = Eigen_Space(LM)
    
    # indicator = Indicators[:, 28]
    # p_coord, p_indicator = project_one_indicator(evecs, evals, indicator)
    # verts = vertexPos.T.copy()
    # tets  = tetFaces.T.copy()
    # ps.init()
    # vol = ps.register_volume_mesh("tet mesh", verts, tets=tets, interior_color=(0.9,0.9,0.9))
    # vol.set_transparency(0.2)
    # vol.add_scalar_quantity('wr', indicator.flatten(), defined_on='vertices', cmap='viridis', enabled=True)
    # vol.add_scalar_quantity('wrp', p_indicator.flatten(), defined_on='vertices', cmap='viridis', enabled=True)
    # ps.show()
    
    p_coords = project_indicators(evecs, evals, Indicators)
    # using Euclidean distance in the projected space
    Dist_2 = np.full((N, N), 2.0+1e-8) - 2*(p_coords.T@p_coords)
    Dist = np.sqrt(Dist_2)
    # using Cosine distance in the projected space
    # C = p_coords.T @ p_coords
    # Dist = np.arccos(np.clip(C, -1.0, 1.0))

    # cluster_selection_epsilon: 0.5-0.8 is a stable state 
    hdb = HDBSCAN(metric='precomputed', 
                  min_cluster_size=min_cluster_size,
                  min_samples=min_samples,
                  cluster_selection_epsilon=cluster_selection_epsilon,
                  cluster_selection_method=cluster_selection_method,
                  )
    hdb.fit(Dist)
    labels = hdb.labels_
    return labels

def Graph_Clustering(indicators):
    # A more straight-forward method:
    # finding the connected component of the graph constructed by the region
    Indicators = csr_matrix(indicators.astype(bool))      # make it sparse & boolean

    # Project to N×N co-occurrence (counts of shared 1s)
    Graph = (Indicators.T @ Indicators).tocsr()                        # integer counts on edges

    # Convert to unweighted adjacency by thresholding > 0, drop self-loops
    Graph.data = (Graph.data > 0).astype(np.int8)
    Graph.setdiag(0); Graph.eliminate_zeros()

    # Undirected connected components
    n_comp, labels = connected_components(Graph, directed=False)
    return labels


if __name__ == "__main__":
    filename = 'data/test.mat'
    data_dict = load_matlab(filename)

    params = data_dict['parameters']
    weak_regions = data_dict['weak_regions']
    V = params['V']
    numEigenModes = params['numEigenModes']
    numWeakRegions = params['numWeakRegions']
    N = numWeakRegions

    vertexPos = data_dict['mesh_info']['vertexPos']
    tetFaces = data_dict['mesh_info']['tetFaces']

    indicators = np.zeros((V, N))
    for i in range(N):
        WRIds = weak_regions['All_WRIds'][i]
        indicators[WRIds, i] = 1.0

    labels = HDBSCAN_Clustering(indicators, vertexPos, tetFaces)
    # labels = Graph_Clustering(indicators)

    n_labels = labels.max()+1
    flag_labels = np.sum(labels==-1)>0

    cluster_ids, clusters = get_cluster_region(labels, weak_regions['All_WRIds'])

    # visualization of the clustering result
    ps.init()

    mesh_info = data_dict['mesh_info']
    bVertices = mesh_info['bVertices']
    bFacesVIds = mesh_info['bFacesVIds']
    mapping = mesh_info['mapping']
    bV = params['bV']
    mesh = ps.register_surface_mesh("WCSA chair mesh", bVertices.T, bFacesVIds.T)
    for i in range(n_labels+flag_labels):
        idx = i - flag_labels
        color = get_color_quantity(mapping, clusters[idx], bV, i)
        mesh.add_color_quantity(f"weak region {idx}", color, defined_on='vertices', enabled=False)

    ps.show()