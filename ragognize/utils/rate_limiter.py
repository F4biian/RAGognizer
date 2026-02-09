import time
import threading
from collections import deque

class RateLimiter:
    def __init__(self, limits):
        self.rules = []
        self.lock = threading.Lock()

        for interval_str, max_calls in limits.items():
            interval = self._parse_interval(interval_str)
            self.rules.append({
                'interval': interval,
                'limit': max_calls,
                'calls': deque()
            })

    def _parse_interval(self, interval_str):
        unit = interval_str[-1]
        try:
            value = float(interval_str[:-1])
        except ValueError:
            raise ValueError(f"Invalid interval format: {interval_str}")

        multipliers = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
        if unit not in multipliers:
            raise ValueError(f"Unsupported time unit: {unit} in {interval_str}")

        return value * multipliers[unit]

    def acquire(self):
        while True:
            now = time.time()
            wait_times = []

            with self.lock:
                for rule in self.rules:
                    calls = rule['calls']
                    interval = rule['interval']

                    while calls and calls[0] <= now - interval:
                        calls.popleft()

                    if len(calls) >= rule['limit']:
                        wait_times.append(calls[0] + interval - now)

            if not wait_times:
                break
            
            time.sleep(max(wait_times))

        with self.lock:
            for rule in self.rules:
                rule['calls'].append(time.time())
