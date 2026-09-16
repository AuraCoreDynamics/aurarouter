import json
import pytest
from pathlib import Path
from aurarouter.trace_logger import TraceLogger

def test_trace_logger_logs_and_reads(tmp_path):
    trace_dir = tmp_path / "traces"
    logger = TraceLogger(trace_dir=trace_dir)
    
    session_id = "test-session-123"
    
    logger.log_stage(session_id, "intent_classification", {"intent": "test"})
    logger.log_stage(session_id, "circuit_breaker_trip", {"model_id": "ollama"})
    
    records = logger.read_trace(session_id)
    assert len(records) == 2
    assert records[0]["stage_type"] == "intent_classification"
    assert records[0]["payload"]["intent"] == "test"
    assert records[1]["stage_type"] == "circuit_breaker_trip"
    assert records[1]["payload"]["model_id"] == "ollama"

def test_trace_logger_empty_session_id(tmp_path):
    trace_dir = tmp_path / "traces"
    logger = TraceLogger(trace_dir=trace_dir)
    
    logger.log_stage("", "test", {})
    records = logger.read_trace("")
    assert len(records) == 0

def test_trace_logger_nonexistent_session(tmp_path):
    trace_dir = tmp_path / "traces"
    logger = TraceLogger(trace_dir=trace_dir)
    
    records = logger.read_trace("nonexistent")
    assert len(records) == 0
