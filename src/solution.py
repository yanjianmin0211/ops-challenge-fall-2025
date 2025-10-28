import polars as pl
import numpy as np
from numba import njit


@njit(fastmath=True, nogil=True)
def rolling_rank_numba(values, window, out):
    n = len(values)
    for i in range(n):
        start = max(0, i - window + 1)
        current_val = values[i]
        
        rank_sum = 0
        for j in range(start, i + 1):
            if values[j] <= current_val:
                rank_sum += 1
        
        window_size = i - start + 1
        out[i] = rank_sum / window_size
    
    return out


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    def compute_rolling_rank(s: pl.Series) -> pl.Series:
        values = s.to_numpy()
        result = np.empty(len(values), dtype=np.float32)
        rolling_rank_numba(values, window, result)
        return pl.Series(result)
    
    df = (
        pl.scan_parquet(input_path)
        .select([
            pl.col('symbol'),
            pl.col('Close').cast(pl.Float32)
        ])
        .with_columns(
            pl.col('Close')
            .map_batches(compute_rolling_rank)
            .over('symbol')
            .alias('rank')
        )
        .select('rank')
    ).collect()
    
    return df.to_numpy()
