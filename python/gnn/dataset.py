import pickle
from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset
import itertools

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
    tetFaces = data_dict['mesh_info']['tetFaces'].T
    bvIndices = data_dict['mesh_info']['bvIndices'].T
    p_all = data_dict['results']['p_all'].T
    stress_all = data_dict['results']['stress_all'].T
    
    vertexPos = torch.tensor(vertexPos, device=device).float()
    tetFaces = torch.tensor(tetFaces, device=device)
    bvIndices = torch.tensor(bvIndices, device=device).long().squeeze()
    p_all = torch.tensor(p_all, device=device).float()
    stress_all = torch.tensor(stress_all, device=device)
    bvFeatures = torch.zeros(V, 1, device=device)
    bvFeatures[bvIndices, 0] = 1.0
    p_label = torch.zeros(V, 75, device=device)
    p_label[bvIndices] = p_all
    stress_max = torch.max(stress_all, dim=0, keepdim=True)[0]
    
    return {
        'vertexPos': vertexPos,        # (V, 3)
        'tetFaces': tetFaces,          # (T, 4)
        'bvFeatures': bvFeatures,      # (V, 1)
        'p_label': p_label,            # (V, 75)
        'stress_max': stress_max,      # (1, 75)
    }

def tet_to_edges(tetFaces, num_nodes):
    edge_set = set()
    for tet in tetFaces:
        for i, j in itertools.combinations(tet, 2):
            edge_set.add((int(i), int(j)))
            edge_set.add((int(j), int(i)))
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    return edge_index
    
def build_data(data_dict):
    vertexPos = data_dict['vertexPos']
    tetFaces = data_dict['tetFaces']
    bvFeatures = data_dict['bvFeatures']
    p_label = data_dict['p_label']
    stress_max = data_dict['stress_max']
    
    num_nodes = vertexPos.size(0)
    edge_index = tet_to_edges(tetFaces, num_nodes)
    
    x = torch.cat([vertexPos, bvFeatures], dim=1)
    
    data = Data(x=x, edge_index=edge_index, y=p_label, stress_max=stress_max)
    return data
    
class TetMeshDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []  # Not used since we load files directly
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
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
    root = Path("~/Documents/phy/python/data/chair_40").expanduser()
    pattern = "*/data.pkl"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # files = root.glob(pattern)
    # for f in files:
    #     load_pkl(f, device)
    dataset = TetMeshDataset(root)