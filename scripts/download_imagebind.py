import argparse
import os
import requests
from pathlib import Path

def download_file(url: str, target_path: Path):
    print(f"[INFO] Downloading {url} to {target_path}...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"[INFO] Successfully downloaded {url} to {target_path}.")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Error downloading {url}: {e}")
        print("[ERROR] Please ensure the URL is correct and you have internet access.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ImageBind model.")
    parser.add_argument("--url", type=str, default="https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth", help="The URL to download the ImageBind model from.")
    parser.add_argument("--target-dir", type=str, default="ckpt/pretrained_ckpt/imagebind_ckpt/huge", help="The local directory to save the model files.")
    args = parser.parse_args()

    file_name = Path(args.url).name
    target_path = Path(args.target_dir) / file_name
    
    try:
        import requests
    except ImportError:
        print("requests library is not installed. Please install it using: pip install requests")
        exit(1)

    download_file(args.url, target_path)
