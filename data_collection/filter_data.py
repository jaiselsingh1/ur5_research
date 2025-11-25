# post processing script to filter trajectories generated post training 
import numpy as np
from pathlib import Path 
from typing import Tuple 
from jaxtyping import Float
import shutil 

def filter_trajectory(
    obj_pos: Float[np.ndarray, "T 3"], 
    ee_pos: Float[np.ndarray, "T 3"] = None,
    target_pos: Float[np.ndarray, "3"] = None,
    verbose: bool = False
) -> Tuple[bool, str]:
    x, y, z = obj_pos[:, 0], obj_pos[:, 1], obj_pos[:, 2]
    
    if np.any(z < -0.14):
        if verbose:
            print("Object has fallen")
        return False, "dropped"
    
    if np.any((x < 0.20) | (x > 0.80) | (y < -0.59) | (y > 0.59)):
        if verbose:
            print("Object is off table")
        return False, "off_table"
    
    total_displacement = np.linalg.norm(obj_pos[-1] - obj_pos[0])
    if total_displacement < 0.02:
        if verbose:
            print(f"Object barely moved: {total_displacement:.4f}m")
        return False, "stuck"
    
    if ee_pos is not None:
        ee_to_obj = np.linalg.norm(ee_pos - obj_pos, axis=1)
        if np.any(ee_to_obj > 0.8):
            if verbose:
                print("EE wandered too far from object")
            return False, "ee_far"
    
    if target_pos is not None:
        final_dist = np.linalg.norm(obj_pos[-1] - target_pos)
        if final_dist > 0.10:
            if verbose:
                print(f"Didn't reach target: {final_dist:.4f}m away")
            return False, "incomplete"
    
    return True, "ok"


def filter_dataset(
    data_dir: Path, 
    output_dir: Path,
    verbose: bool = True
) -> dict:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"ok": 0, "dropped": 0, "off_table": 0, "stuck": 0, "ee_far": 0, "incomplete": 0}
    valid_idx = 1
    
    trajectory_dirs = sorted(data_dir.glob("traj_*"))
    
    if len(trajectory_dirs) == 0:
        print(f"No trajectory folders found in {data_dir}")
        return stats
    
    for traj_dir in trajectory_dirs:
        obj_pos_file = traj_dir / "obj_pos_states.npz"
        if not obj_pos_file.exists():
            print(f"Warning: {obj_pos_file} not found, skipping")
            continue
            
        obj_data = np.load(obj_pos_file)
        obj_pos = obj_data[obj_data.files[0]]
        
        ee_pos = None
        ee_pos_file = traj_dir / "ee_pos_states.npz"
        if ee_pos_file.exists():
            ee_data = np.load(ee_pos_file)
            ee_pos = ee_data[ee_data.files[0]]
        
        target_pos = None
        
        is_valid, reason = filter_trajectory(obj_pos, ee_pos, target_pos, verbose=False)
        stats[reason] += 1
        
        if is_valid:
            output_traj_dir = output_dir / f"traj_{valid_idx:04d}"
            shutil.copytree(traj_dir, output_traj_dir)
            valid_idx += 1
            if verbose:
                print(f"{traj_dir.name} -> {output_traj_dir.name}")
        elif verbose:
            print(f"{traj_dir.name}: filtered out ({reason})")
    
    print(f"\nFiltering complete:")
    print(f"  Valid: {stats['ok']} / {sum(stats.values())}")
    for reason, count in stats.items():
        if reason != "ok" and count > 0:
            print(f"  {reason}: {count}")
    
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    filter_dataset(Path(args.input), Path(args.output))