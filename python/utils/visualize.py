# This utility file provides functions to visualize tetrahedron meshes
# In different ways like 2D surface mesh, and 3D volume mesh
# Also, it provides functions to visualize stress, forces,
# weakregions, and clusters on the mesh
import numpy as np
import matplotlib.pyplot as plt

OKABE_ITO_HEX = [
    "#000000", "#E69F00", "#56B4E9", "#009E73", 
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]
more = ["#E41A1C","#1F77B4","#2CA02C","#9467BD",
        "#FF7F0E","#17BECF","#8C564B","#BCBD22",
        "#7F7F7F","#AEC7E8","#98DF8A","#FF9896"]
OKABE_ITO_HEX = OKABE_ITO_HEX + more

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
    rgb = np.array([r, g, b])
    return rgb

background_color = (0.3, 0.3, 0.3)   # Dark gray

def get_color_quantity(mapping, region, bV, color_id):
    rgb = hex_to_rgb(OKABE_ITO_HEX[color_id % len(OKABE_ITO_HEX)])
    bregion = mapping[region]
    bregion = bregion[bregion != -1]
    color = np.tile(background_color, (bV, 1))
    color[bregion, :] = np.tile(rgb, (bregion.shape[0], 1))
    return color

def get_cluster_region(labels, All_WRIds):
    n_labels = labels.max()+1
    flag_labels = np.sum(labels==-1)>0
    cluster_ids = [i-flag_labels for i in range(n_labels+flag_labels)]
    clusters = {cid: np.empty(0, dtype=int) for cid in cluster_ids}
    for i in range(len(All_WRIds)):
        region = All_WRIds[i].flatten()
        label = labels[i]
        clusters[label] = np.unique(np.concatenate([clusters[label], region]))
    return cluster_ids, clusters

def draw_distance_matrix(Dist):
    """ Visualize the pairwise distance matrix using matplotlib """
    plt.figure(figsize=(8,6))
    im = plt.imshow(Dist, origin="lower", interpolation="nearest")
    plt.title("Pairwise distance matrix")
    plt.xlabel("vector index")
    plt.ylabel("vector index")
    plt.colorbar(im, label="distance")
    plt.tight_layout()
    plt.show()
