from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class TaskResult:
    task_id: str
    model_response: str
    is_correct: Optional[bool]  # None if not auto-gradable
    latency_ms: float
    tokens_generated: int
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class BenchmarkRun:
    run_id: str
    config: Dict[str, Any]  # BenchmarkConfig as dict
    results: list[TaskResult]
    timestamp: str
    total_duration_sec: float
    
    @classmethod
    def create_new(cls, config):
        return cls(
            run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            results=[],
            timestamp=datetime.now().isoformat(),
            total_duration_sec=0.0
        )
