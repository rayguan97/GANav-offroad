import os, glob, shutil
from pathlib import Path
from tqdm import tqdm
import re 


dir_lst = ["./data/rellis/image/", "./data/rellis/annotation/"]
remove_folders = ["pylon_camera_node/", "pylon_camera_node_label_id/"]
for d, r in zip(dir_lst, remove_folders):
    move_lst = []
    for f in tqdm(list(Path(d).rglob("*.*"))):
        move_lst.append(f)
    for f in tqdm(move_lst):
        shutil.move(f, re.sub(r, "", str(f)))

    for remove_d in (Path(d).rglob("*" + r)):
        assert len(os.listdir(remove_d)) == 0
        os.rmdir(remove_d)

