# This utility function loads MATLAB computed data from .mat files
# and organizes them into a structured Python dictionary for further processing.
from scipy.io import loadmat
import numpy as np

def load_matlab(filename):
    data = loadmat(filename)
    OPT = data['OPT']
    Tet = data['Tet']

    # Tetrahedron mesh basic info
    vertexPos = Tet['vertexPoss'][0, 0][:3, :]     # size[3, V]
    tetFaces = Tet['tetFaces'][0, 0] - 1           # size[4, T]
    bvIndices = Tet['boundaryVertexIndices'][0, 0] 
    bvIndices = bvIndices - 1                      # size[1, bV]       
    bFacesVIds = Tet['boundaryFaceVIds'][0, 0] - 1 # size[3, bF]
    bFacesTIds = Tet['boundaryFaceTIds'][0, 0] - 1 # size[1, bF]

    V = vertexPos.shape[1]
    T = tetFaces.shape[1]
    bV = bvIndices.shape[1]
    bF = bFacesVIds.shape[1]
    numEigenModes = 15
    numWeakRegions = 5
    
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
    All_WRIds = []
    All_bWRIds = []
    for eid in range(numEigenModes):
        for wid in range(numWeakRegions):
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
            'bFacesVIds': bFacesVIds,
            'bVertices': bVertices,
            'bFacesTIds': bFacesTIds,
            'mapping': mapping,
        },
        'weak_regions': {
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
    }
    return data_dict
