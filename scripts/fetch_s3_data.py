#!/usr/bin/env python3
"""
S3에서 Polymarket crypto trades 데이터 다운로드 및 스키마 확인
"""

import os
from pathlib import Path

import boto3
import pandas as pd

BUCKET = os.getenv("S3_BUCKET", "an-trading-research")
PREFIX = "polymarket"
DATA_DIR = Path(__file__).parent.parent / "data" / "s3_cache"


def list_s3_recursive(s3, prefix: str) -> list[dict]:
    """S3 경로 아래 모든 파일 목록"""
    files = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            files.append({
                "key": obj["Key"],
                "size_mb": obj["Size"] / 1024 / 1024,
                "name": obj["Key"].split("/")[-1]
            })

    return files


def download_file(s3, key: str, local_dir: Path) -> Path:
    """S3 파일 다운로드"""
    # 경로 구조 유지
    relative_path = key.replace(f"{PREFIX}/", "")
    local_path = local_dir / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        print(f"  [SKIP] Already exists: {local_path}")
        return local_path

    print(f"  Downloading: {key} -> {local_path}")
    s3.download_file(BUCKET, key, str(local_path))
    return local_path


def main():
    s3 = boto3.client("s3")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"S3 Bucket: s3://{BUCKET}/{PREFIX}/")
    print("=" * 70)

    # 1. 먼저 전체 구조 확인
    print("\n[1] Listing all files...")
    all_files = list_s3_recursive(s3, PREFIX)

    print(f"\nTotal files: {len(all_files)}")
    print(f"Total size: {sum(f['size_mb'] for f in all_files):.1f} MB")

    # 구조별로 그룹핑
    print("\n[2] File structure:")
    by_prefix = {}
    for f in all_files:
        parts = f["key"].split("/")
        if len(parts) >= 2:
            group = "/".join(parts[:3]) if len(parts) >= 3 else "/".join(parts[:2])
            if group not in by_prefix:
                by_prefix[group] = []
            by_prefix[group].append(f)

    for prefix, files in sorted(by_prefix.items()):
        total_mb = sum(f["size_mb"] for f in files)
        print(f"  {prefix}: {len(files)} files, {total_mb:.1f} MB")

    # 2. crypto 관련 파일만 필터링
    print("\n[3] Crypto trades files:")
    crypto_files = [f for f in all_files if "crypto" in f["key"].lower() or "1h" in f["key"]]

    if not crypto_files:
        print("  No crypto files found. Listing all trades files:")
        crypto_files = [f for f in all_files if "trades" in f["key"]]

    for f in crypto_files[:20]:  # 처음 20개만
        print(f"  {f['key']:<60} {f['size_mb']:>8.2f} MB")

    if len(crypto_files) > 20:
        print(f"  ... and {len(crypto_files) - 20} more files")

    # 3. 샘플 파일 다운로드 (작은 것부터)
    print("\n[4] Downloading sample files for schema inspection...")

    # metadata 먼저
    metadata_files = [f for f in all_files if "metadata" in f["key"] and f["key"].endswith(".parquet")]
    for f in metadata_files[:3]:
        download_file(s3, f["key"], DATA_DIR)

    # 가장 작은 trades 파일
    trades_files = [f for f in all_files if "trades" in f["key"] and f["key"].endswith(".parquet")]
    trades_files.sort(key=lambda x: x["size_mb"])

    sample_files = []
    for f in trades_files[:5]:  # 가장 작은 5개
        local_path = download_file(s3, f["key"], DATA_DIR)
        sample_files.append(local_path)

    # 4. 스키마 확인
    print("\n[5] Inspecting schemas...")

    for local_path in sample_files:
        if local_path.exists():
            print(f"\n{'=' * 70}")
            print(f"File: {local_path}")
            print("=" * 70)

            try:
                df = pd.read_parquet(local_path)
                print(f"Shape: {df.shape}")
                print(f"\nColumns: {list(df.columns)}")
                print(f"\nDtypes:\n{df.dtypes}")
                print(f"\nSample (first 3 rows):")
                print(df.head(3).to_string())

                # 시간 관련 컬럼 확인
                time_cols = [c for c in df.columns if any(t in c.lower() for t in ["time", "date", "ts"])]
                if time_cols:
                    print(f"\nTime columns: {time_cols}")
                    for col in time_cols:
                        print(f"  {col}: min={df[col].min()}, max={df[col].max()}")

            except Exception as e:
                print(f"Error reading: {e}")

    print("\n" + "=" * 70)
    print("Done! Check the output above for schema details.")
    print(f"Downloaded files are in: {DATA_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
