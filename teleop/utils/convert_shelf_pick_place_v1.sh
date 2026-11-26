#!/usr/bin/env bash
set -euo pipefail
base=/home/unitree/g1d_dataset
name=lerobot_shelf_pick_place_v1
[[ ! -e "$base/$name" ]] || { echo "Output already exists"; exit 1; }
tools=/home/unitree/zhoumy/xr_teleoperate/unitree_lerobot
export PATH="/home/unitree/miniconda3/envs/unitree_lerobot/bin:$PATH"
export PYTHONPATH="$tools:$tools/unitree_lerobot/lerobot/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_LEROBOT_HOME
HF_LEROBOT_HOME=$(mktemp -d "$base/.shelf-build-XXXXXX")
echo "Staging $HF_LEROBOT_HOME"
python -u - <<'PY'
import os,json,tempfile
from pathlib import Path
from unitree_lerobot.utils.convert_unitree_json_to_lerobot import create_empty_dataset,populate_dataset,DatasetConfig
import pyarrow.parquet as pq
from unitree_lerobot.utils.constants import ROBOT_CONFIGS
# Shelf recordings: color_2 is the right wrist and color_3 is the left wrist.
ROBOT_CONFIGS['Unitree_G1_Dex1'].camera_to_image_key.update(
    color_2='cam_right_wrist', color_3='cam_left_wrist',
)
stage=Path(os.environ['HF_LEROBOT_HOME'])
base=stage.parent
name='lerobot_shelf_pick_place_v1'
groups=['shelf_pick_place_zmy_green','shelf_pick_place_blue']
dst=stage/name
ds=create_empty_dataset(name,'Unitree_G1_Dex1',dataset_config=DatasetConfig(image_writer_processes=0,image_writer_threads=4))
lengths=[]; goals=[]
for group in groups:
    for f in sorted((base/group).glob('episode_*/data.json')):
        with f.open() as handle: doc=json.load(handle)
        n=len(doc['data']); goal=doc['text']['goal']
        assert n and goal.strip() and doc['info']['image']['fps']==30
        for row in doc['data']:
            assert set(row['colors'])=={f'color_{i}' for i in range(4)}
        del doc
        with tempfile.TemporaryDirectory(dir=stage) as t:
            task=Path(t)/group;task.mkdir();(task/f.parent.name).symlink_to(f.parent)
            populate_dataset(ds,Path(t),'Unitree_G1_Dex1')
        lengths.append(n);goals.append(goal)
        print('Saved',len(lengths),group,f.parent.name,n,flush=True)
ds.finalize()
info=json.loads((dst/'meta/info.json').read_text())
assert info['codebase_version']=='v3.0' and info['total_frames']==sum(lengths) and info['total_episodes']==len(lengths) and info['total_tasks']==len(set(goals))
eps=pq.read_table(dst/'meta/episodes').to_pandas().sort_values('episode_index')
assert list(eps['length'])==lengths
assert list(eps['dataset_to_index'])==list(__import__('itertools').accumulate(lengths))
for key,feature in info['features'].items():
    if feature['dtype']=='video':
        for _,r in eps.iterrows():
            v=dst/info['video_path'].format(video_key=key,chunk_index=int(r[f'videos/{key}/chunk_index']),file_index=int(r[f'videos/{key}/file_index']))
            assert v.is_file() and abs((r[f'videos/{key}/to_timestamp']-r[f'videos/{key}/from_timestamp'])*30-r['length'])<1
out=base/name
if out.exists(): raise FileExistsError(out)
dst.rename(out)
print('VERIFIED',out,len(lengths),sum(lengths),set(goals))
PY
