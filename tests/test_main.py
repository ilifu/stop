import pytest
import asyncio
import json
import sys
import os
from unittest.mock import AsyncMock, patch
import polars as pl

from slurm import (
    get_slurm_data,
    process_partition_summary,
    process_job_summaries,
    process_node_summary,
    process_pending_job_waiting_times_summary,
    format_seconds_to_human_readable
)

# Mock data for Slurm commands
MOCK_SINFO_JSON = {
    "sinfo": [
        {"partition": {"name": "debug"}, "nodes": {"total": 1, "idle": 1, "allocated": 0}, "cpus": {"total": 4, "idle": 4, "allocated": 0}},
        {"partition": {"name": "normal"}, "nodes": {"total": 2, "idle": 1, "allocated": 1}, "cpus": {"total": 8, "idle": 4, "allocated": 4}},
    ]
}

MOCK_SQUEUE_JSON = {
    "jobs": [
        {"account": "test_account_1", "job_state": ["RUNNING"], "user_name": "user1", "eligible_time": {"number": 1678886400}, "start_time": {"number": 1678886400}},
        {"account": "test_account_1", "job_state": ["PENDING"], "user_name": "user1", "eligible_time": {"number": 1678886000}, "start_time": {"number": 1678886400}},
        {"account": "test_account_2", "job_state": ["RUNNING"], "user_name": "user2", "eligible_time": {"number": 1678886400}, "start_time": {"number": 1678886400}},
        {"account": "test_account_2", "job_state": ["PENDING"], "user_name": "user2", "eligible_time": {"number": 1678885000}, "start_time": {"number": 1678886400}},
        {"account": "test_account_1", "job_state": ["COMPLETED"], "user_name": "user1", "eligible_time": {"number": 0}, "start_time": {"number": 0}},
    ]
}

MOCK_SCONTROL_JSON = {
    "nodes": [
        {"cpus": 4, "real_memory": 16000, "alloc_cpus": 0, "alloc_memory": 0, "state": ["IDLE"]},
        {"cpus": 8, "real_memory": 32000, "alloc_cpus": 4, "alloc_memory": 16000, "state": ["ALLOCATED"]},
        {"cpus": 4, "real_memory": 16000, "alloc_cpus": 0, "alloc_memory": 0, "state": ["DRAINED"]},
        {"cpus": 8, "real_memory": 32000, "alloc_cpus": 4, "alloc_memory": 16000, "state": ["RESERVATION", "ALLOCATED"]},
        {"cpus": 8, "real_memory": 32000, "alloc_cpus": 2, "alloc_memory": 8000, "state": ["MIXED"]}, # Mixed node
    ]
}

@pytest.fixture
def mock_subprocess_shell():
    with patch('asyncio.create_subprocess_shell') as mock_create_subprocess_shell:
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_create_subprocess_shell.return_value = mock_process
        yield mock_create_subprocess_shell, mock_process

@pytest.mark.asyncio
async def test_get_slurm_data_success(mock_subprocess_shell):
    mock_create_subprocess_shell, mock_process = mock_subprocess_shell
    mock_process.communicate.return_value = (json.dumps(MOCK_SINFO_JSON).encode(), b'')
    
    data = await get_slurm_data("sinfo --json")
    assert data == MOCK_SINFO_JSON
    mock_create_subprocess_shell.assert_called_once_with(
        "sinfo --json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

@pytest.mark.asyncio
async def test_get_slurm_data_command_not_found(mock_subprocess_shell, capsys):
    mock_create_subprocess_shell, mock_process = mock_subprocess_shell
    mock_create_subprocess_shell.side_effect = FileNotFoundError
    
    data = await get_slurm_data("nonexistent_command --json")
    assert data is None
    captured = capsys.readouterr()
    assert "Error: Command 'nonexistent_command' not found." in captured.out

@pytest.mark.asyncio
async def test_get_slurm_data_json_decode_error(mock_subprocess_shell, capsys):
    mock_create_subprocess_shell, mock_process = mock_subprocess_shell
    mock_process.communicate.return_value = (b'invalid json', b'')
    
    data = await get_slurm_data("sinfo --json")
    assert data is None
    captured = capsys.readouterr()
    assert "Error: Could not decode JSON from sinfo --json output." in captured.out

@pytest.mark.asyncio
async def test_get_slurm_data_non_zero_return_code(mock_subprocess_shell, capsys):
    mock_create_subprocess_shell, mock_process = mock_subprocess_shell
    mock_process.returncode = 1
    mock_process.communicate.return_value = (b'', b'Slurm error message')
    
    data = await get_slurm_data("sinfo --json")
    assert data is None
    captured = capsys.readouterr()
    assert "Error running sinfo --json: Slurm error message" in captured.out

@pytest.mark.asyncio
async def test_process_partition_summary(mock_subprocess_shell):
    mock_create_subprocess_shell, mock_process = mock_subprocess_shell
    mock_process.communicate.return_value = (json.dumps(MOCK_SINFO_JSON).encode(), b'')
    
    summary_df = process_partition_summary(MOCK_SINFO_JSON)
    assert summary_df is not None
    assert "Partition" in summary_df.columns
    assert "Total Nodes" in summary_df.columns
    assert "Idle Nodes" in summary_df.columns
    assert "Allocated Nodes" in summary_df.columns
    assert "Total CPUs" in summary_df.columns
    assert "Free CPUs" in summary_df.columns
    assert "Allocated CPUs" in summary_df.columns
    
    # Verify data
    debug_row = summary_df.filter(pl.col("Partition") == "debug")
    assert debug_row["Total Nodes"].item() == 1
    assert debug_row["Idle Nodes"].item() == 1
    assert debug_row["Allocated Nodes"].item() == 0
    assert debug_row["Total CPUs"].item() == 4
    assert debug_row["Free CPUs"].item() == 4
    assert debug_row["Allocated CPUs"].item() == 0

    normal_row = summary_df.filter(pl.col("Partition") == "normal")
    assert normal_row["Total Nodes"].item() == 2
    assert normal_row["Idle Nodes"].item() == 1
    assert normal_row["Allocated Nodes"].item() == 1
    assert normal_row["Total CPUs"].item() == 8
    assert normal_row["Free CPUs"].item() == 4
    assert normal_row["Allocated CPUs"].item() == 4

@pytest.mark.asyncio
async def test_process_job_summaries(mock_subprocess_shell):
    mock_create_subprocess_shell, mock_process = mock_subprocess_shell
    mock_process.communicate.return_value = (json.dumps(MOCK_SQUEUE_JSON).encode(), b'')
    
    account_summary_df, user_summary_df = process_job_summaries(MOCK_SQUEUE_JSON)
    
    assert account_summary_df is not None
    assert user_summary_df is not None
    
    # Verify account summary
    assert "account" in account_summary_df.columns
    assert "Total Jobs" in account_summary_df.columns
    assert "RUNNING" in account_summary_df.columns
    assert "PENDING" in account_summary_df.columns
    assert "COMPLETED" in account_summary_df.columns

    acc1_row = account_summary_df.filter(pl.col("account") == "test_account_1")
    assert acc1_row["Total Jobs"].item() == 3
    assert acc1_row["RUNNING"].item() == 1
    assert acc1_row["PENDING"].item() == 1
    assert acc1_row["COMPLETED"].item() == 1

    acc2_row = account_summary_df.filter(pl.col("account") == "test_account_2")
    assert acc2_row.shape[0] == 1
    assert acc2_row["Total Jobs"].item() == 2
    assert acc2_row["RUNNING"].item() == 1
    assert acc2_row["PENDING"].item() == 1
    assert "COMPLETED" not in acc2_row.columns or acc2_row["COMPLETED"].item() == 0

    # Verify user summary
    assert "user_name" in user_summary_df.columns
    assert "Total Jobs" in user_summary_df.columns
    assert "RUNNING" in user_summary_df.columns
    assert "PENDING" in user_summary_df.columns
    assert "COMPLETED" in user_summary_df.columns

    user1_row = user_summary_df.filter(pl.col("user_name") == "user1")
    assert user1_row.shape[0] == 1
    assert user1_row["Total Jobs"].item() == 3
    assert user1_row["RUNNING"].item() == 1
    assert user1_row["PENDING"].item() == 1
    assert user1_row["COMPLETED"].item() == 1

    user2_row = user_summary_df.filter(pl.col("user_name") == "user2")
    assert user2_row.shape[0] == 1
    assert user2_row["Total Jobs"].item() == 2
    assert user2_row["RUNNING"].item() == 1
    assert user2_row["PENDING"].item() == 1
    assert "COMPLETED" not in user2_row.columns or user2_row["COMPLETED"].item() == 0

def test_process_job_summaries_no_jobs():
    account_summary_df, user_summary_df = process_job_summaries({"jobs": []})
    
    assert account_summary_df is not None
    assert user_summary_df is not None
    
    assert account_summary_df.is_empty()
    assert user_summary_df.is_empty()

@pytest.mark.asyncio
async def test_process_node_summary(mock_subprocess_shell):
    mock_create_subprocess_shell, mock_process = mock_subprocess_shell
    mock_process.communicate.return_value = (json.dumps(MOCK_SCONTROL_JSON).encode(), b'')
    
    summary_df = process_node_summary(MOCK_SCONTROL_JSON)
    
    assert summary_df is not None
    assert "Metric" in summary_df.columns
    assert "Value" in summary_df.columns
    
    metrics = summary_df.select(pl.col("Metric")).to_series().to_list()
    values = summary_df.select(pl.col("Value")).to_series().to_list()

    expected_metrics = [
        "Total Nodes", "Total CPUs", "Total Memory (MB)",
        "Allocated CPUs", "Allocated Memory (MB)",
        "Broken Nodes", "Nodes in Reservation", "Mixed Nodes"
    ]
    expected_values = [
        5, # Total Nodes
        32, # Total CPUs
        128000, # Total Memory
        10, # Allocated CPUs
        40000, # Allocated Memory
        1, # Broken Nodes (DRAINED)
        1, # Nodes in Reservation
        3 # Mixed Nodes
    ]

    for metric, expected_value in zip(expected_metrics, expected_values):
        assert summary_df.filter(pl.col("Metric") == metric)["Value"].item() == expected_value

@pytest.mark.asyncio
async def test_process_pending_job_waiting_times_summary(mock_subprocess_shell):
    mock_create_subprocess_shell, mock_process = mock_subprocess_shell
    mock_process.communicate.return_value = (json.dumps(MOCK_SQUEUE_JSON).encode(), b'')
    
    summary_df = process_pending_job_waiting_times_summary(MOCK_SQUEUE_JSON)
    
    assert summary_df is not None
    assert "Metric" in summary_df.columns
    assert "Value" in summary_df.columns

    # Expected waiting times for pending jobs:
    # user1: 1678886400 - 1678886000 = 400 seconds
    # user2: 1678886400 - 1678885000 = 1400 seconds
    # Max: 1400, Median: (400+1400)/2 = 900, Mean: (400+1400)/2 = 900

    assert summary_df.filter(pl.col("Metric") == "Max Waiting Time")["Value"].item() == "0d 0h 23m 20s" # 1400 seconds
    assert summary_df.filter(pl.col("Metric") == "Median Waiting Time")["Value"].item() == "0d 0h 15m 0s" # 900 seconds
    assert summary_df.filter(pl.col("Metric") == "Mean Waiting Time")["Value"].item() == "0d 0h 15m 0s" # 900 seconds

@pytest.mark.asyncio
async def test_process_pending_job_waiting_times_summary_no_pending_jobs(mock_subprocess_shell):
    mock_create_subprocess_shell, mock_process = mock_subprocess_shell
    mock_process.communicate.return_value = (json.dumps({"jobs": []}).encode(), b'')
    
    summary_df = process_pending_job_waiting_times_summary({"jobs": []})
    
    assert summary_df is not None
    assert summary_df.shape == (3, 2)
    assert "Metric" in summary_df.columns
    assert "Value" in summary_df.columns
    assert summary_df.filter(pl.col("Metric") == "Max Waiting Time")["Value"].item() == "0d 0h 0m 0s"
    assert summary_df.filter(pl.col("Metric") == "Median Waiting Time")["Value"].item() == "0d 0h 0m 0s"
    assert summary_df.filter(pl.col("Metric") == "Mean Waiting Time")["Value"].item() == "0d 0h 0m 0s"


def test_format_seconds_to_human_readable():
    assert format_seconds_to_human_readable(0) == "0d 0h 0m 0s"
    assert format_seconds_to_human_readable(60) == "0d 0h 1m 0s"
    assert format_seconds_to_human_readable(3600) == "0d 1h 0m 0s"
    assert format_seconds_to_human_readable(86400) == "1d 0h 0m 0s"
    assert format_seconds_to_human_readable(90061) == "1d 1h 1m 1s"
    assert format_seconds_to_human_readable(None) == "N/A"