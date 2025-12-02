import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub is not installed. Run: pip install huggingface-hub")
    sys.exit(1)


def find_files(root: Path, patterns: List[str]) -> List[Path]:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(root.rglob(pattern))
    return matches


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def standardized_rename_in_place(folder: Path):
    """
    Rename files to dia{dia_id}utt{utt_id}.{ext} to match training loader expectations.
    This mirrors the logic in merg_code/dataset/preprocess_dataset.py.
    """
    for file in folder.glob("*.*"):
        name = file.name
        if name.endswith(".wav") or name.endswith(".mp4"):
            # expect original like: dia00001utt0x.wav/mp4 → convert to dia1uttx
            try:
                dia_num = name[3:8].lstrip('0')
                utt_num = str(int(name[11]) + 1)
                new_name = f"dia{dia_num}utt{utt_num}{file.suffix}"
                new_path = file.with_name(new_name)
                if new_path != file:
                    file.rename(new_path)
                    print(f"Renamed: {name} -> {new_name}")
            except Exception:
                # skip files that don't match expected pattern
                continue


def main():
    parser = argparse.ArgumentParser(description="Download ZhangHanXD/AvaMERG dataset and arrange files for training")
    parser.add_argument("--repo-id", default="ZhangHanXD/AvaMERG", help="Hugging Face dataset repo id")
    parser.add_argument("--target-dir", default="merg_data", help="Directory to place arranged data")
    parser.add_argument("--split", default="train", choices=["train", "test", "validation", "dev", "val"], help="Dataset split to arrange")
    parser.add_argument("--use-cli-layout", action="store_true", help="Keep snapshot layout only (no reorg)")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    split = args.split

    print(f"Downloading dataset {args.repo_id} (repo_type=dataset) ...")
    snapshot_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
    )
    print(f"Snapshot downloaded at: {snapshot_dir}")

    if args.use_cli_layout:
        print("--use-cli-layout specified; skipping file reorganization.")
        return

    # Prepare destination folders
    split_root = target_dir / split
    audio_out = split_root / "audio"
    video_out = split_root / "video"
    ensure_dir(audio_out)
    ensure_dir(video_out)

    # Move audio/video files into expected layout
    audio_files = find_files(snapshot_dir, ["*.wav", "*.flac", "*.mp3"])  # broaden patterns just in case
    video_files = find_files(snapshot_dir, ["*.mp4", "*.avi", "*.mov"])   # broaden patterns

    moved_audio = 0
    for f in audio_files:
        dest = audio_out / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved_audio += 1

    moved_video = 0
    for f in video_files:
        dest = video_out / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved_video += 1

    # Standardize names to dia{dia_id}utt{utt_id}.ext where possible
    standardized_rename_in_place(audio_out)
    standardized_rename_in_place(video_out)

    # Locate and place split JSON
    # Training loader expects: merg_data/<mode>.json (e.g., merg_data/train.json)
    json_candidates = find_files(snapshot_dir, ["*.json"])
    placed_json = False
    for jf in json_candidates:
        # prefer exact split filename
        if jf.name.lower() in {f"{split}.json", f"{split}_split.json"}:
            dest = target_dir / f"{split}.json"
            if not jf.samefile(dest):
                shutil.copy2(str(jf), str(dest))
            placed_json = True
            break
    if not placed_json and json_candidates:
        # fallback: take the first json and name it split.json
        dest = target_dir / f"{split}.json"
        shutil.copy2(str(json_candidates[0]), str(dest))
        placed_json = True

    # Summary
    print("\nDownload and organization summary:")
    print(f"- Audio files moved: {moved_audio}")
    print(f"- Video files moved: {moved_video}")
    print(f"- JSON placed: {'yes' if placed_json else 'no'} -> {target_dir / f'{split}.json'}")
    print(f"- Expected training paths ready:")
    print(f"  audio_path: {audio_out}")
    print(f"  video_path: {video_out}")
    print("\nYou can now run training, e.g.:")
    print(
        "deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28459 "
        "merg_code/train.py --mode train --audio_path merg_data/train/audio "
        "--video_path merg_data/train/video --save_path ckpt/merg_ckpt --log_path ./logs --max_length 512"
    )


if __name__ == "__main__":
    main()