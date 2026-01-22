#!/usr/bin/env python3
"""
15분 마켓 데이터 전처리 스크립트

각 파일이 30분 단위로 겹쳐서 저장되어 있는 raw 데이터를
15분 마켓(slug) 단위로 묶어서 정리합니다.

입력: data/datalake/timebased/crypto/updown/15m/{symbol}/...
출력: data/datalake/processed/15m/{symbol}/{date}/{slug}/{type}.parquet.gz
"""

import gzip
import io
from pathlib import Path
from collections import defaultdict
import pandas as pd
import argparse


BASE_DIR = Path("data/datalake/timebased/crypto/updown/15m")
OUTPUT_DIR = Path("data/datalake/processed/15m")
SYMBOLS = ["btc", "eth", "sol", "xrp"]
DATA_TYPES = ["book", "price_change", "last_trade_price", "tick_size_change"]
MARKET_DURATION_SEC = 900  # 15분
MIN_COVERAGE_MINS = 13  # 커버리지 경고 기준


def read_parquet_gz(file_path: Path) -> pd.DataFrame:
    """gzip 압축된 parquet 파일 읽기"""
    with gzip.open(file_path, "rb") as gz:
        return pd.read_parquet(io.BytesIO(gz.read()))


def save_parquet_gz(df: pd.DataFrame, file_path: Path):
    """gzip 압축된 parquet 파일 저장"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, compression="gzip", index=False)


def extract_market_timestamp(slug: str) -> int | None:
    """slug에서 마켓 시작 타임스탬프 추출
    예: btc-updown-15m-1768875300 -> 1768875300
    """
    try:
        ts = int(slug.split("-")[-1])
        # 유효한 타임스탬프 범위 확인 (2025-2027)
        if 1700000000 < ts < 1900000000:
            return ts
        return None
    except (ValueError, IndexError):
        return None


def process_symbol(symbol: str, verbose: bool = True) -> dict:
    """심볼별 데이터 전처리"""
    input_dir = BASE_DIR / symbol
    output_dir = OUTPUT_DIR / symbol

    if not input_dir.exists():
        print(f"[{symbol.upper()}] 입력 디렉토리 없음: {input_dir}")
        return {"symbol": symbol, "markets": 0, "alerts": []}

    # 데이터 타입별로 slug 데이터 수집
    # slug_data[slug][data_type] = [df1, df2, ...]
    slug_data = defaultdict(lambda: defaultdict(list))
    file_counts = {}

    for dtype in DATA_TYPES:
        files = sorted(input_dir.rglob(f"{dtype}_*.parquet.gz"))
        file_counts[dtype] = len(files)

        for f in files:
            try:
                df = read_parquet_gz(f)

                # slug별로 분리
                for slug in df["slug"].unique():
                    if not slug.startswith(f"{symbol}-updown-15m-"):
                        continue
                    if extract_market_timestamp(slug) is None:
                        continue

                    slug_df = df[df["slug"] == slug].copy()
                    slug_data[slug][dtype].append(slug_df)
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] {f.name}: {e}")

    if verbose:
        print(f"[{symbol.upper()}] 파일 수: {file_counts}")
        print(f"[{symbol.upper()}] 발견된 마켓: {len(slug_data)}개")

    # slug별 전처리 및 저장
    stats = {"symbol": symbol, "markets": 0, "alerts": []}

    for slug in sorted(slug_data.keys()):
        market_ts = extract_market_timestamp(slug)
        market_start = pd.to_datetime(market_ts, unit="s")
        market_end = pd.to_datetime(market_ts + MARKET_DURATION_SEC, unit="s")
        date_str = market_start.strftime("%Y/%m/%d")

        slug_output_dir = output_dir / date_str / slug
        slug_stats = {"slug": slug, "market_start": market_start}

        for dtype, dfs in slug_data[slug].items():
            if not dfs:
                continue

            # 데이터 병합
            combined = pd.concat(dfs, ignore_index=True)

            # 타임스탬프 변환
            combined["ts_dt"] = pd.to_datetime(combined["ts"], unit="ms")

            # 마켓 기간 필터링 (정확히 15분)
            mask = (combined["ts_dt"] >= market_start) & (combined["ts_dt"] <= market_end)
            filtered = combined[mask].copy()

            if len(filtered) == 0:
                continue

            # 중복 제거 (hash + ts 기준)
            if "hash" in filtered.columns:
                filtered = filtered.drop_duplicates(subset=["hash", "ts"])
            else:
                filtered = filtered.drop_duplicates(subset=["ts"])

            # 시간순 정렬
            filtered = filtered.sort_values("ts_dt").reset_index(drop=True)

            # 저장
            out_path = slug_output_dir / f"{dtype}.parquet.gz"
            save_parquet_gz(filtered, out_path)

            slug_stats[dtype] = len(filtered)

            # book 데이터로 커버리지 계산
            if dtype == "book" and len(filtered) > 0:
                coverage_start = filtered["ts_dt"].min()
                coverage_end = filtered["ts_dt"].max()
                coverage_mins = (coverage_end - coverage_start).total_seconds() / 60
                slug_stats["coverage_mins"] = coverage_mins

                if coverage_mins < MIN_COVERAGE_MINS:
                    alert = {
                        "slug": slug,
                        "market_start": market_start.strftime("%Y-%m-%d %H:%M"),
                        "coverage_mins": round(coverage_mins, 1),
                        "rows": len(filtered),
                    }
                    stats["alerts"].append(alert)

        stats["markets"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="15분 마켓 데이터 전처리")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS, help="처리할 심볼")
    parser.add_argument("-v", "--verbose", action="store_true", help="상세 출력")
    args = parser.parse_args()

    print("=" * 70)
    print("15분 마켓 데이터 전처리")
    print(f"입력: {BASE_DIR}")
    print(f"출력: {OUTPUT_DIR}")
    print(f"데이터 타입: {DATA_TYPES}")
    print("=" * 70)

    all_stats = []
    all_alerts = []

    for sym in args.symbols:
        stats = process_symbol(sym, args.verbose)
        all_stats.append(stats)
        all_alerts.extend(stats["alerts"])
        print()

    # 최종 통계
    print("=" * 70)
    print("전처리 완료")
    print("-" * 70)

    total_markets = sum(s["markets"] for s in all_stats)
    for s in all_stats:
        print(f"  {s['symbol'].upper()}: {s['markets']} 마켓")

    print("-" * 70)
    print(f"  TOTAL: {total_markets} 마켓")
    print(f"  출력: {OUTPUT_DIR}")
    print("=" * 70)

    # 커버리지 부족 Alert
    if all_alerts:
        print()
        print("!" * 70)
        print(f"ALERT: 커버리지 부족 마켓 ({len(all_alerts)}개)")
        print(f"기준: {MIN_COVERAGE_MINS}분 미만")
        print("-" * 70)
        for a in sorted(all_alerts, key=lambda x: x["coverage_mins"]):
            print(f"  {a['slug']}: {a['coverage_mins']}분 ({a['rows']} rows) @ {a['market_start']}")
        print("!" * 70)


if __name__ == "__main__":
    main()
