import pickle
from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset
import itertools
import os

def load_pkl(file_path, device):
    """Load a pickle file and return its content."""
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)
    # data_dict = {
    #     'mesh_info': {
    #         'vertexPos': vertexPos,
    #         'tetFaces': tetFaces,
    #         'bvIndices': bvIndices,
    #         'bvNorms': bvNorms,
    #         'bvAreas': bvAreas,
    #         'bFacesVIds': bFacesVIds,
    #         'bVertices': bVertices,
    #         'bFacesTIds': bFacesTIds,
    #         'mapping': mapping,
    #     },
    #     'weak_regions': {
    #         'eigenvectors': eigenvectors,
    #         'All_WRIds': All_WRIds,
    #         'All_bWRIds': All_bWRIds,
    #     },
    #     'parameters': {
    #         'V': V,
    #         'T': T,
    #         'bV': bV,
    #         'bF': bF,
    #         'numEigenModes': numEigenModes,
    #         'numWeakRegions': numWeakRegions,
    #     },
    #     'results': {
    #         'best_eid': best_eid,
    #         'best_wid': best_wid,
    #         'p_all': p,
    #         'u_all': u,
    #         'stress_all': s,
    #     }
    # }
    
    V = data_dict['parameters']['V']
    vertexPos = data_dict['mesh_info']['vertexPos'].T
    bvIndices = data_dict['mesh_info']['bvIndices'].T
    p_all = data_dict['results']['p_all'].T
    stress_all = data_dict['results']['stress_all'].T
    bvNorms = data_dict['mesh_info']['bvNorms'].T
    bFacesVIds = data_dict['mesh_info']['bFacesVIds'].T
    
    vertexPos = torch.tensor(vertexPos, device=device).float()
    bvIndices = torch.tensor(bvIndices, device=device).long().squeeze()
    p_all = torch.tensor(p_all, device=device).float()
    stress_all = torch.tensor(stress_all, device=device)
    bvNorms = torch.tensor(bvNorms, device=device).float()
    bFacesVIds = torch.tensor(bFacesVIds, device=device).int()
    vertexPos = vertexPos[bvIndices, :]
    stress_max = torch.max(stress_all, dim=0, keepdim=True)[0]
    
    return {
        'vertexPos': vertexPos,        # (bV, 3)
        'p_label': p_all,              # (bV, 75)
        'bvNorms': bvNorms,            # (bV, 3)
        'stress_max': stress_max,      # (1, 75)
        'bFacesVIds': bFacesVIds,      # (bF, 3)
    }

def build_data(data_dict):
    position = data_dict['vertexPos']
    label = data_dict['p_label']
    label_threshold = 0.5
    label = (label >= label_threshold).float()
    feature = data_dict['bvNorms']
    tris = data_dict['bFacesVIds']
    
    edges = torch.cat([
        tris[:, [0, 1]],
        tris[:, [1, 2]],
        tris[:, [2, 0]],
        tris[:, [1, 0]],
        tris[:, [2, 1]],
        tris[:, [0, 2]],
    ], dim=0)  # shape: (6*T, 2)
    edges = torch.unique(edges, dim=0).T  # shape: (2, E)

    data = Data(x=feature, y=label, pos=position, edge_index=edges)
    return data

class PointNetDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []  # Not used since we load files directly
    
    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'pointnet')

    def process(self):
        data_list = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for file_path in Path(self.root).glob("*/data.pkl"):
            print(f"Processing {file_path}")
            data_dict = load_pkl(file_path, device)
            data = build_data(data_dict)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
if __name__ == "__main__":
    root = Path("~/Documents/phy/python/data/chair_600").expanduser()
    pattern = "*/data.pkl"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # files = root.glob(pattern)
    # for f in files:
    #     load_pkl(f, device)
    dataset = PointNetDataset(root)