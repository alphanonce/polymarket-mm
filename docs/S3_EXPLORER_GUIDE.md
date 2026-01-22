# S3 Explorer 사용 가이드

Data Lake S3 Explorer를 통해 Polymarket 시장 데이터를 탐색하는 방법을 설명합니다.

---

## 접속 정보

| 항목 | 값 |
|------|-----|
| URL | `http://52.213.17.123:9094` |
| 인증 | Basic Auth (ID/PW 별도 공유) |
| 접근 조건 | VPN 연결 필수 |

---

## 폴더 구조

**핵심 원칙**: `global/market_info/`와 `timebased/`는 **동일한 하위 구조**를 가집니다.

```
data/polymarket/
│
├── global/                                      # 마켓 메타데이터
│   │
│   ├── market_info/                             # 마켓 정보 (outcome, close_price 포함)
│   │   └── crypto/
│   │       ├── updown/                          # Up/Down 마켓
│   │       │   ├── 5m/{asset}/YYYY/MM/DD/       # btc, eth, sol, xrp
│   │       │   ├── 15m/{asset}/YYYY/MM/DD/
│   │       │   ├── 1h/{asset}/YYYY/MM/DD/       # bitcoin, ethereum, solana, xrp
│   │       │   ├── 4h/{asset}/YYYY/MM/DD/
│   │       │   ├── daily/{asset}/YYYY/MM/DD/
│   │       │   └── weekly/{asset}/YYYY/MM/DD/
│   │       │
│   │       ├── above/{asset}/YYYY/MM/DD/        # "XX above YYk" 마켓
│   │       ├── price-on/{asset}/YYYY/MM/DD/     # "price on date" 마켓
│   │       ├── price-target/{asset}/YYYY/MM/DD/ # "what price will hit" 마켓
│   │       └── markets/YYYY/MM/DD/              # fallback (미분류)
│   │
│   ├── downtime/YYYY/MM/DD/                     # 서비스 다운타임 기록
│   └── rtds_crypto_prices/YYYY/MM/DD/           # 실시간 암호화폐 가격
│
├── timebased/                                   # 시계열 데이터 (30분 배치)
│   └── crypto/
│       ├── updown/                              # (market_info와 동일 구조)
│       │   ├── 5m/{asset}/YYYY/MM/DD/
│       │   ├── 15m/{asset}/YYYY/MM/DD/
│       │   ├── 1h/{asset}/YYYY/MM/DD/
│       │   ├── 4h/{asset}/YYYY/MM/DD/
│       │   ├── daily/{asset}/YYYY/MM/DD/
│       │   └── weekly/{asset}/YYYY/MM/DD/
│       │
│       ├── above/{asset}/YYYY/MM/DD/
│       ├── price-on/{asset}/YYYY/MM/DD/
│       └── price-target/{asset}/YYYY/MM/DD/
│
└── markets/{slug}/YYYY/MM/DD/                   # 일반 마켓 (fallback)
```

### Asset 명명 규칙

| Timeframe | Asset 예시 | 설명 |
|-----------|-----------|------|
| 5m, 15m, 4h | `btc`, `eth`, `sol`, `xrp` | 소문자 약어 |
| 1h, daily, weekly | `bitcoin`, `ethereum`, `solana`, `xrp` | 풀네임 |
| above, price-on | `bitcoin`, `ethereum` | 풀네임 |

---

## 파일 명명 규칙

### Timebased 데이터 (orderbook 등)

```
{event_type}_{slug}_{minEpoch}-{maxEpoch}_{startTime}-{endTime}.parquet.gz

예시:
orderbook_delta_btc-updown-15m_1768665600-1768666500_14-00-00-14-30-00.parquet.gz
│               │              │                      │
│               │              │                      └─ flush 시간 범위 (HH-MM-SS)
│               │              └─ epoch 범위 (포함된 마켓 인스턴스들)
│               └─ Normalized Slug
└─ 이벤트 타입
```

### Market Info

```
market_info_{slug}_{time}.parquet.gz

예시:
market_info_btc-updown-15m_14-30-00-123456789.parquet.gz
```

### 파일명 구성 요소

| 구성 요소 | 설명 | 예시 |
|----------|------|------|
| `event_type` | 이벤트 종류 | `orderbook_delta`, `orderbook_snapshot`, `last_trade`, `market_info` |
| `slug` | 정규화된 마켓 식별자 | `btc-updown-15m`, `bitcoin-above` |
| `minEpoch-maxEpoch` | 파일에 포함된 마켓 epoch 범위 | `1768665600-1768666500` |
| `time` | flush 시간 | `14-30-00` (UTC) |

---

## Slug 패턴

### 원본 → 정규화 변환

| 마켓 유형 | Original Slug | Normalized Slug | 경로 |
|----------|--------------|-----------------|------|
| 15분 | `btc-updown-15m-1768665600` | `btc-updown-15m` | `updown/15m/btc/` |
| 1시간 | `bitcoin-up-or-down-january-17-9am-et` | `bitcoin-updown-1h` | `updown/1h/bitcoin/` |
| 4시간 | `btc-updown-4h-1768665600` | `btc-updown-4h` | `updown/4h/btc/` |
| Daily | `bitcoin-up-or-down-on-january-17` | `bitcoin-updown-daily` | `updown/daily/bitcoin/` |
| Above | `bitcoin-above-100k-on-january-18` | `bitcoin-above` | `above/bitcoin/` |
| Price-On | `will-the-price-of-xrp-be-3-or-higher` | `xrp-price-on` | `price-on/xrp/` |

### Epoch 범위의 의미

파일명의 `minEpoch-maxEpoch`는 해당 파일에 포함된 **마켓 인스턴스들의 시작 시간 범위**입니다.

```
예: btc-updown-15m_1768665600-1768666500_14-00-00.parquet.gz

- 1768665600 = 2026-01-17 14:00:00 UTC에 시작한 15분 마켓
- 1768666500 = 2026-01-17 14:15:00 UTC에 시작한 15분 마켓
- 이 파일에는 2개의 15분 마켓 데이터가 포함됨
```

---

## 데이터 컬럼

### orderbook_delta / orderbook_snapshot (오더북)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `ts` | int64 | 이벤트 타임스탬프 (ms) |
| `local_ts` | int64 | 로컬 수신 시간 (ms) |
| `slug` | string | 마켓 슬러그 |
| `condition_id` | string | 조건 ID |
| `token_id` | string | 토큰 ID |
| `bids` | array | 매수 호가 `[[price, size], ...]` |
| `asks` | array | 매도 호가 `[[price, size], ...]` |
| `hash` | string | 오더북 해시 |

### market_info (마켓 정보)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `ts` | int64 | 타임스탬프 (ms) |
| `local_ts` | int64 | 로컬 수신 시간 (ms) |
| `slug` | string | 원본 마켓 슬러그 |
| `series_slug` | string | 시리즈 슬러그 (정규화됨) |
| `condition_id` | string | 조건 ID |
| `token_id_yes` | string | YES 토큰 ID |
| `token_id_no` | string | NO 토큰 ID |
| `symbol` | string | 심볼 (BTC, ETH 등) |
| `timeframe` | string | 타임프레임 (15m, 1h 등) |
| `start_ts` | int64 | 마켓 시작 시간 (ms) |
| `end_ts` | int64 | 마켓 종료 시간 (ms) |
| `status` | string | **`active`** (구독 시) / **`resolved`** (종료 시) |
| `outcome` | string | 결과: **`up`** / **`down`** (resolved일 때만) |
| `close_price` | float64 | 종료 가격 (resolved일 때만) |

#### status 필드 설명

- **`active`**: 마켓 구독 시작 시점에 기록
- **`resolved`**: 마켓 종료 후 outcome 확정 시 기록
- 하나의 마켓에 대해 **2개의 레코드**가 생성됨 (active → resolved)

---

## 데이터 수집 주기

| 데이터 | 주기 | 설명 |
|--------|------|------|
| 오더북 (timebased) | **30분 배치** | 30분마다 S3에 flush |
| market_info (global) | **이벤트 기반** | active(구독 시), resolved(종료 시) |

> **참고**: 배치 경계와 마켓 lifecycle이 다를 수 있음. `slug` 기준으로 JOIN하여 분석.

---

## Explorer 사용 팁

### 정렬
- **Name** 클릭 → 이름순 정렬
- **Size** 클릭 → 크기순 정렬
- **Modified** 클릭 → 수정일순 정렬
- 같은 컬럼 재클릭 → 오름차순/내림차순 토글
- 정렬 설정은 **자동 저장**됨 (새로고침해도 유지)

### 필터
- 상단 검색창에 키워드 입력 → 실시간 필터링
- 예: `btc` 입력 → BTC 관련 파일만 표시

### 다중 선택 다운로드
1. 파일 왼쪽 체크박스 선택
2. 우하단 "Download N files" 버튼 클릭
3. 선택한 파일들이 개별 다운로드됨

### Parquet 미리보기
- `.parquet.gz` 파일의 **Preview** 버튼 클릭
- 최대 100행까지 테이블 형태로 미리보기
- 20MB 이하 파일만 지원

### URL 공유
- 현재 폴더 경로가 URL에 반영됨
- 예: `http://52.213.17.123:9094/data/polymarket/timebased/crypto/updown/15m/btc/`
- URL 복사해서 팀원과 공유 가능

---

## 자주 사용하는 경로

### Timebased (오더북)

| 용도 | 경로 |
|------|------|
| BTC 15분 오더북 | `/data/polymarket/timebased/crypto/updown/15m/btc/` |
| ETH 15분 오더북 | `/data/polymarket/timebased/crypto/updown/15m/eth/` |
| BTC 1시간 오더북 | `/data/polymarket/timebased/crypto/updown/1h/bitcoin/` |
| BTC Above 오더북 | `/data/polymarket/timebased/crypto/above/bitcoin/` |

### Global (마켓 정보)

| 용도 | 경로 |
|------|------|
| BTC 15분 market_info | `/data/polymarket/global/market_info/crypto/updown/15m/btc/` |
| BTC 1시간 market_info | `/data/polymarket/global/market_info/crypto/updown/1h/bitcoin/` |
| 실시간 가격 | `/data/polymarket/global/rtds_crypto_prices/` |

---

## 분석 예시 (SQL)

> **가정**: DuckDB, Athena, Spark SQL 등에서 Parquet 파일을 로드한 상태.
> 한 달치 데이터를 다운로드하여 로컬에서 분석하는 시나리오.

### 핵심 개념: slug 종류

| 필드 | 출처 | 예시 | 용도 |
|------|------|------|------|
| `slug` (orderbook) | CLOB WebSocket | `btc-updown-15m-1768705200` | **JOIN 키** |
| `slug` (market_info) | CLOB WebSocket | `btc-updown-15m-1768705200` | **JOIN 키** (동일) |
| `series_slug` | Gamma API | `btc-up-or-down-15m` | Series 수준 참조 |
| (S3 경로) | Writer 정규화 | `btc-updown-15m` | 폴더 라우팅 |

> **주의**: `series_slug`는 Gamma API 형식(`btc-up-or-down-15m`)이며,
> S3 폴더 경로의 slug(`btc-updown-15m`)와 다릅니다.

**JOIN은 항상 `slug` (original) 기준으로 수행**합니다.

---

### 데이터 로드 (DuckDB 예시)

```sql
-- BTC 15분 마켓 (한 달치)
CREATE TABLE orderbook_15m AS
SELECT * FROM read_parquet('timebased/crypto/updown/15m/btc/2026/01/*/*.parquet.gz');

CREATE TABLE market_info_15m AS
SELECT * FROM read_parquet('global/market_info/crypto/updown/15m/btc/2026/01/*/*.parquet.gz');

-- BTC 1시간 마켓 (한 달치) - 주의: asset이 'bitcoin'
CREATE TABLE orderbook_1h AS
SELECT * FROM read_parquet('timebased/crypto/updown/1h/bitcoin/2026/01/*/*.parquet.gz');

CREATE TABLE market_info_1h AS
SELECT * FROM read_parquet('global/market_info/crypto/updown/1h/bitcoin/2026/01/*/*.parquet.gz');
```

---

### 1. 마켓별 outcome/close_price 조회

모든 resolved 마켓의 결과 확인:

```sql
SELECT
    slug,
    symbol,
    timeframe,
    from_unixtime(start_ts/1000) AS start_time,
    from_unixtime(end_ts/1000) AS end_time,
    outcome,
    close_price
FROM market_info_15m
WHERE status = 'resolved'
ORDER BY end_ts DESC
LIMIT 100;
```

---

### 2. 마켓 Lifecycle (active → resolved) 연결

하나의 마켓에 대해 active/resolved 두 레코드를 하나로 합침:

```sql
WITH lifecycle AS (
    SELECT
        slug,
        symbol,
        timeframe,
        start_ts,
        end_ts,
        MAX(CASE WHEN status = 'active' THEN ts END) AS subscribed_at,
        MAX(CASE WHEN status = 'resolved' THEN ts END) AS resolved_at,
        MAX(CASE WHEN status = 'resolved' THEN outcome END) AS outcome,
        MAX(CASE WHEN status = 'resolved' THEN close_price END) AS close_price
    FROM market_info_15m
    GROUP BY slug, symbol, timeframe, start_ts, end_ts
)
SELECT
    slug,
    from_unixtime(start_ts/1000) AS market_start,
    from_unixtime(end_ts/1000) AS market_end,
    from_unixtime(subscribed_at/1000) AS subscribed_at,
    from_unixtime(resolved_at/1000) AS resolved_at,
    outcome,
    close_price
FROM lifecycle
WHERE outcome IS NOT NULL
ORDER BY start_ts DESC;
```

---

### 3. 오더북과 market_info JOIN (마켓별 outcome 매핑)

**핵심**: timebased의 `slug`와 market_info의 `slug`는 동일 (original slug).

```sql
-- 각 오더북 레코드에 해당 마켓의 outcome 매핑
WITH resolved_markets AS (
    SELECT
        slug,
        outcome,
        close_price,
        start_ts,
        end_ts
    FROM market_info_15m
    WHERE status = 'resolved'
)
SELECT
    ob.ts,
    from_unixtime(ob.ts/1000) AS event_time,
    ob.slug,
    ob.token_id,
    ob.bids,
    ob.asks,
    rm.outcome,
    rm.close_price
FROM orderbook_15m ob
JOIN resolved_markets rm ON ob.slug = rm.slug
WHERE ob.ts BETWEEN rm.start_ts AND rm.end_ts  -- 마켓 기간 내 데이터만
ORDER BY ob.ts
LIMIT 1000;
```

---

### 4. 마켓 종료 직전 스프레드 분석

마켓 종료 5분 전 best bid/ask 스프레드:

```sql
WITH resolved_markets AS (
    SELECT slug, outcome, close_price, start_ts, end_ts
    FROM market_info_15m
    WHERE status = 'resolved'
),
last_snapshots AS (
    SELECT
        ob.slug,
        ob.ts,
        ob.bids,
        ob.asks,
        rm.outcome,
        rm.close_price,
        rm.end_ts,
        ROW_NUMBER() OVER (PARTITION BY ob.slug ORDER BY ob.ts DESC) AS rn
    FROM orderbook_15m ob
    JOIN resolved_markets rm ON ob.slug = rm.slug
    WHERE ob.ts BETWEEN (rm.end_ts - 5*60*1000) AND rm.end_ts  -- 종료 5분 전
)
SELECT
    slug,
    from_unixtime(ts/1000) AS snapshot_time,
    -- Best bid/ask 추출 (JSON 배열 첫 번째 요소)
    CAST(json_extract(bids, '$[0][0]') AS DOUBLE) AS best_bid,
    CAST(json_extract(asks, '$[0][0]') AS DOUBLE) AS best_ask,
    CAST(json_extract(asks, '$[0][0]') AS DOUBLE) - CAST(json_extract(bids, '$[0][0]') AS DOUBLE) AS spread,
    outcome,
    close_price
FROM last_snapshots
WHERE rn = 1  -- 각 마켓의 마지막 스냅샷만
ORDER BY ts DESC;
```

---

### 5. Up/Down 승률 통계

한 달간 BTC 15분 마켓의 Up/Down 비율:

```sql
SELECT
    outcome,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
    ROUND(AVG(close_price), 2) AS avg_close_price
FROM market_info_15m
WHERE status = 'resolved'
  AND outcome IS NOT NULL
GROUP BY outcome;
```

---

### 6. 시간대별 outcome 패턴 분석

UTC 시간대별 Up/Down 경향:

```sql
SELECT
    EXTRACT(HOUR FROM from_unixtime(start_ts/1000)) AS hour_utc,
    outcome,
    COUNT(*) AS count,
    ROUND(AVG(close_price), 4) AS avg_close_price
FROM market_info_15m
WHERE status = 'resolved'
  AND outcome IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, 2;
```

---

### 7. 오더북 깊이 변화 추적

특정 마켓의 오더북 depth(유동성) 변화:

```sql
WITH orderbook_depth AS (
    SELECT
        slug,
        ts,
        from_unixtime(ts/1000) AS event_time,
        -- 상위 5개 레벨 총 사이즈 계산
        (
            SELECT SUM(CAST(json_extract(level, '$[1]') AS DOUBLE))
            FROM json_each(bids) WITH ORDINALITY AS t(level, idx)
            WHERE idx <= 5
        ) AS bid_depth_5,
        (
            SELECT SUM(CAST(json_extract(level, '$[1]') AS DOUBLE))
            FROM json_each(asks) WITH ORDINALITY AS t(level, idx)
            WHERE idx <= 5
        ) AS ask_depth_5
    FROM orderbook_15m
    WHERE slug = 'btc-updown-15m-1768705200'  -- 특정 마켓
)
SELECT * FROM orderbook_depth
ORDER BY ts;
```

---

### 8. 1시간 마켓 (hourly) 분석

1시간 마켓도 동일한 패턴으로 분석 (asset명만 다름):

```sql
-- 1시간 마켓 lifecycle
WITH hourly_lifecycle AS (
    SELECT
        slug,
        symbol,
        start_ts,
        end_ts,
        MAX(CASE WHEN status = 'resolved' THEN outcome END) AS outcome,
        MAX(CASE WHEN status = 'resolved' THEN close_price END) AS close_price
    FROM market_info_1h
    GROUP BY slug, symbol, start_ts, end_ts
)
SELECT
    slug,
    symbol,
    from_unixtime(start_ts/1000) AS market_start,
    from_unixtime(end_ts/1000) AS market_end,
    outcome,
    close_price,
    -- 마켓 지속 시간 계산
    (end_ts - start_ts) / 1000 / 60 AS duration_minutes
FROM hourly_lifecycle
WHERE outcome IS NOT NULL
ORDER BY start_ts DESC;
```

---

### 9. 15분 vs 1시간 마켓 비교

같은 기간 동안 두 타임프레임의 outcome 비교:

```sql
WITH m15 AS (
    SELECT
        DATE(from_unixtime(start_ts/1000)) AS date,
        outcome,
        COUNT(*) AS cnt
    FROM market_info_15m
    WHERE status = 'resolved' AND outcome IS NOT NULL
    GROUP BY 1, 2
),
m1h AS (
    SELECT
        DATE(from_unixtime(start_ts/1000)) AS date,
        outcome,
        COUNT(*) AS cnt
    FROM market_info_1h
    WHERE status = 'resolved' AND outcome IS NOT NULL
    GROUP BY 1, 2
)
SELECT
    COALESCE(m15.date, m1h.date) AS date,
    m15.outcome AS outcome_15m,
    m15.cnt AS count_15m,
    m1h.outcome AS outcome_1h,
    m1h.cnt AS count_1h
FROM m15
FULL OUTER JOIN m1h ON m15.date = m1h.date AND m15.outcome = m1h.outcome
ORDER BY date, outcome_15m;
```

---

### 10. 전체 분석 파이프라인 (종합)

한 달치 데이터로 마켓별 요약 리포트 생성:

```sql
WITH market_summary AS (
    -- market_info에서 기본 정보
    SELECT
        m.slug,
        m.symbol,
        m.timeframe,
        m.start_ts,
        m.end_ts,
        m.outcome,
        m.close_price
    FROM market_info_15m m
    WHERE m.status = 'resolved'
      AND m.outcome IS NOT NULL
),
orderbook_stats AS (
    -- 오더북에서 통계
    SELECT
        ob.slug,
        COUNT(*) AS orderbook_updates,
        MIN(ob.ts) AS first_update,
        MAX(ob.ts) AS last_update,
        -- 평균 스프레드 (best ask - best bid)
        AVG(
            CAST(json_extract(ob.asks, '$[0][0]') AS DOUBLE) -
            CAST(json_extract(ob.bids, '$[0][0]') AS DOUBLE)
        ) AS avg_spread
    FROM orderbook_15m ob
    GROUP BY ob.slug
)
SELECT
    ms.slug,
    ms.symbol,
    ms.timeframe,
    from_unixtime(ms.start_ts/1000) AS market_start,
    from_unixtime(ms.end_ts/1000) AS market_end,
    ms.outcome,
    ms.close_price,
    os.orderbook_updates,
    ROUND(os.avg_spread, 4) AS avg_spread,
    -- 데이터 커버리지 (첫 업데이트 ~ 마지막 업데이트)
    ROUND((os.last_update - os.first_update) * 100.0 / (ms.end_ts - ms.start_ts), 1) AS coverage_pct
FROM market_summary ms
LEFT JOIN orderbook_stats os ON ms.slug = os.slug
ORDER BY ms.start_ts DESC;
```

---

## 문제 해결

### "No files or folders found"
- S3 접근 권한 문제일 수 있음
- 관리자에게 문의

### 파일 미리보기 실패
- 파일 크기가 20MB 초과
- 손상된 Parquet 파일
- Download 후 로컬에서 확인

### 접속 안 됨
- VPN 연결 상태 확인
- `http://` (https 아님) 확인
- 포트 `9094` 확인

---

## 연락처

문의사항은 Slack `#data-lake` 채널로 연락 바랍니다.
