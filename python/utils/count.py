from pathlib import Path
from utils.load_matlab import *

root = Path("~/Documents/Physical-stability/main/python/data/chair_1200").expanduser()
pattern = "*/tet/WCSA.mat"

shape_ids_with_WCSA = []
good_shapes = []

for model_filename in sorted(root.rglob(pattern)):
    try:
        relative_path = model_filename.relative_to(root)
        shape_id = relative_path.parts[0]
        shape_ids_with_WCSA.append(shape_id)
        data_dict = load_matlab(model_filename)
        V = data_dict['parameters']['V']
        T = data_dict['parameters']['T']
        numWeakRegions = data_dict['parameters']['numWeakRegions']
        if numWeakRegions == 75:
            good_shapes.append(shape_id)
        else:
            print(f"bad shape {shape_id}")

    except Exception as e:
        print(f"Error processing {model_filename}: {e}")

# 把结果写入 root/shape_ids.txt
# output_file = root / "shape_ids.txt"
# with open(output_file, "w") as f:
#     for sid in shape_ids_with_tetmesh:
#         f.write(sid + "\n")

print(f"Done. Found {len(shape_ids_with_WCSA)} shape_ids")
print(f"Found {len(good_shapes)} shapes with exactly 75 weak regions")