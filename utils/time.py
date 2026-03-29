from typing import Dict, List, Tuple

from tqdm import tqdm
from loguru import logger
import time


def time_measurement(stats_dict: Dict[str, float], operation_name: str):
    class TimerContext:
        def __enter__(self):
            logger.debug(f"Doing {operation_name}...")
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            stats_dict[operation_name] = time.perf_counter() - self.start_time
            logger.debug(f"Done {operation_name} in {stats_dict[operation_name]:.3f} seconds")

    return TimerContext()


def time_summarize(perf_stats: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    all_stats = {}

    for perf_stat in tqdm(perf_stats, desc="Calculating performance summary", unit="question", leave=False, delay=1):
        for key, value in perf_stat.items():
            if key not in all_stats:
                all_stats[key] = []
            all_stats[key].append(value)

    average_stats = {}
    total_stats = {}
    for operation, times in all_stats.items():
        sum_times = sum(times)
        avg_time = sum_times / len(times)
        average_stats[operation] = avg_time
        total_stats[operation] = sum_times

    return average_stats, total_stats
