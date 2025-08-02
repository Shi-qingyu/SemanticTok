import os
import shutil


# root = "data/imagenet/val"

# with open("data/imagenet/val.txt", "r") as f:
#     lines = f.readlines()

# for line in lines:
#     line = line.strip()
#     dirname = line.split("/")[0]
#     dirpath = os.path.join(root, dirname)
#     os.makedirs(dirpath, exist_ok=True)

#     src_img_path = os.path.join(root, line.split("/")[1])
#     dst_img_path = os.path.join(dirpath, line.split("/")[1])
#     shutil.move(src_img_path, dst_img_path)


folder2id = {}

with open("data/imagenet/train.txt", "r") as f:
    lines = f.readlines()

import tqdm

for line in tqdm.tqdm(lines):
    line = line.strip()
    path, id = line.split(" ")
    dirname = path.split("/")[0]
    folder2id[dirname] = id

write = []

with open("data/imagenet/val.txt", "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    folder = line.split("/")[0]
    id = folder2id[folder]
    write.append(line + " " + id)

with open("test.txt", "w") as f:
    f.write("\n".join(write))