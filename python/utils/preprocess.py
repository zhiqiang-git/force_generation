from pathlib import Path
from utils.load_matlab import *
import pickle

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
        data_filename = model_filename.parent / "data.pkl"
        numWeakRegions = data_dict['parameters']['numWeakRegions']
        if numWeakRegions == 75:
            good_shapes.append(shape_id)
            with open(data_filename, 'wb') as f:
                pickle.dump(data_dict, f)
                print(f"Processed and saved data for shape {shape_id} with 75 weak regions.")
        else:
            print(f"bad shape {shape_id}")

    except Exception as e:
        print(f"Error processing {model_filename}: {e}")


print(f"Done. Found {len(shape_ids_with_WCSA)} shape_ids")
print(f"Found {len(good_shapes)} shapes with exactly 75 weak regions")