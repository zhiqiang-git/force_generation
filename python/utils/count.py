import time
from pathlib import Path

root = Path("~/Documents/Physical-stability/main/python/data/chair_1200").expanduser()
pattern = "*/tet/tetmesh.m"

shape_ids_with_tetmesh = []

for model_filename in sorted(root.rglob(pattern)):
    try:
        start_time = time.time()
        relative_path = model_filename.relative_to(root)
        shape_id = relative_path.parts[0]
        shape_ids_with_tetmesh.append(shape_id)

    except Exception as e:
        print(f"Error processing {model_filename}: {e}")

# 把结果写入 root/shape_ids.txt
output_file = root / "shape_ids.txt"
with open(output_file, "w") as f:
    for sid in shape_ids_with_tetmesh:
        f.write(sid + "\n")

print(f"Done. Found {len(shape_ids_with_tetmesh)} shape_ids, saved to {output_file}")
