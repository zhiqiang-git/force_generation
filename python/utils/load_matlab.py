# This utility function loads MATLAB computed data from .mat files
# and organizes them into a structured Python dictionary for further processing.
from scipy.io import loadmat
import numpy as np

def load_matlab(filename):
    data = loadmat(filename)
    Basis = data['Basis']
    OPT = data['OPT']
    Tet = data['Tet']
    best_eid = data['eigenmode_id'].item() - 1
    best_wid = data['weakregion_id'].item() - 1
    p = data['p_all']
    u = data['u_all']
    s = data['stress_all']

    # Tetrahedron mesh basic info
    vertexPos = Tet['vertexPoss'][0, 0][:3, :]     # size[3, V]
    tetFaces = Tet['tetFaces'][0, 0] - 1           # size[4, T]
    bvIndices = Tet['boundaryVertexIndices'][0, 0] 
    bvIndices = bvIndices - 1                      # size[1, bV]       
    bFacesVIds = Tet['boundaryFaceVIds'][0, 0] - 1 # size[3, bF]
    bFacesTIds = Tet['boundaryFaceTIds'][0, 0] - 1 # size[1, bF]
    bvNorms = Tet['vertexNors'][0, 0][:, bvIndices[0]]          # size[3, bV]
    bvAreas = Tet['boundaryVertexArea'][0, 0][:, bvIndices[0]] # size[1, bV]
    
    V = vertexPos.shape[1]
    T = tetFaces.shape[1]
    bV = bvIndices.shape[1]
    bF = bFacesVIds.shape[1]
    numEigenModes = 15
    numWeakRegions = 0
    
    vertexPos = np.array(vertexPos)
    tetFaces = np.array(tetFaces)
    bvIndices = np.array(bvIndices)
    bFacesVIds = np.array(bFacesVIds)
    bvNorms = np.array(bvNorms)

    max_id = tetFaces.max() + 1
    mapping = np.full(max_id, -1, dtype=int)
    mapping[bvIndices] = np.arange(len(bvIndices[0]))
    bFacesVIds = mapping[bFacesVIds]
    bVertices = vertexPos[:, bvIndices[0]]

    # Weak region info
    eigenvectors = Basis['eigenVecs'][0, 0].reshape(V, -1, 15)  # size[V, 3, 15]
    All_WRIds = []
    All_bWRIds = []
    for eid in range(numEigenModes):
        numWRIds = OPT['weakRegions'][0, 0][0, eid].shape[1]
        numWeakRegions += numWRIds
        for wid in range(numWRIds):
            WRIds = OPT['weakRegions'][0, 0][0, eid][0, wid] - 1
            bWRIds = mapping[WRIds]
            bWRIds = bWRIds[bWRIds != -1]
            All_WRIds.append(WRIds)
            All_bWRIds.append(bWRIds)

    # Organize obtained data and return them as a two_level dictionary
    data_dict = {
        'mesh_info': {
            'vertexPos': vertexPos,
            'tetFaces': tetFaces,
            'bvIndices': bvIndices,
            'bvNorms': bvNorms,
            'bvAreas': bvAreas,
            'bFacesVIds': bFacesVIds,
            'bVertices': bVertices,
            'bFacesTIds': bFacesTIds,
            'mapping': mapping,
        },
        'weak_regions': {
            'eigenvectors': eigenvectors,
            'All_WRIds': All_WRIds,
            'All_bWRIds': All_bWRIds,
        },
        'parameters': {
            'V': V,
            'T': T,
            'bV': bV,
            'bF': bF,
            'numEigenModes': numEigenModes,
            'numWeakRegions': numWeakRegions,
        },
        'results': {
            'best_eid': best_eid,
            'best_wid': best_wid,
            'p_all': p,
            'u_all': u,
            'stress_all': s,
        }
    }
    return data_dict
