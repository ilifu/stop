#!/usr/bin/env python3
import json
import asyncio
import subprocess
import polars as pl
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable
from textual.containers import Horizontal, Vertical, ScrollableContainer

async def get_slurm_data(command):
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            print(f"Error running {command}: {stderr.decode().strip()}")
            return None
        return json.loads(stdout.decode().strip())
    except FileNotFoundError:
        print(f"Error: Command '{command.split()[0]}' not found. Is Slurm installed and in your PATH?")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {command} output.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while running {command}: {e}")
        return None

def process_partition_summary(sinfo_data):
    if not sinfo_data:
        print("Failed to retrieve sinfo data for summary.")
        return None

    sinfo_df = pl.DataFrame(sinfo_data["sinfo"])

    # Group sinfo data by partition name and sum relevant columns
    partition_summary_df = sinfo_df.group_by(pl.col("partition").struct.field("name").alias("Partition")).agg([
        pl.col("nodes").struct.field("total").sum().alias("Total Nodes"),
        pl.col("nodes").struct.field("idle").sum().alias("Idle Nodes"),
        pl.col("nodes").struct.field("allocated").sum().alias("Allocated Nodes"),
        pl.col("cpus").struct.field("total").sum().alias("Total CPUs"),
        pl.col("cpus").struct.field("idle").sum().alias("Free CPUs"),
        pl.col("cpus").struct.field("allocated").sum().alias("Allocated CPUs"),
    ])

    return partition_summary_df.sort("Total Nodes", descending=True)

def process_job_summaries(squeue_data):
    if not squeue_data or "jobs" not in squeue_data or not squeue_data["jobs"]:
        print("Failed to retrieve squeue data for job summaries or no jobs found.")
        # Return empty DataFrames with expected schema to prevent ColumnNotFoundError
        account_summary_df = pl.DataFrame(schema={
            "account": pl.String,
            "Total Jobs": pl.UInt32,
            "RUNNING": pl.UInt32,
            "PENDING": pl.UInt32,
            "COMPLETED": pl.UInt32,
        })
        user_summary_df = pl.DataFrame(schema={
            "user_name": pl.String,
            "Total Jobs": pl.UInt32,
            "RUNNING": pl.UInt32,
            "PENDING": pl.UInt32,
            "COMPLETED": pl.UInt32,
        })
        return account_summary_df, user_summary_df

    squeue_df = pl.DataFrame(squeue_data["jobs"])

    # Explode job_state list to count each state individually
    squeue_exploded_df = squeue_df.explode("job_state")

    # Summary by Account
    account_summary_df = squeue_exploded_df.group_by(["account", "job_state"]).agg(
        pl.len().alias("count")
    ).pivot(
        index="account",
        on="job_state",
        values="count",
        aggregate_function="first"
    ).fill_null(0)

    # Add Total Jobs column for account summary
    total_jobs_account = squeue_exploded_df.group_by("account").agg(pl.len().alias("Total Jobs"))
    account_summary_df = account_summary_df.join(total_jobs_account, on="account", how="left")

    # Reorder columns to have Total Jobs first for account summary
    account_summary_df = account_summary_df.select([
        "account", "Total Jobs", pl.exclude("account", "Total Jobs")
    ]).sort("Total Jobs", descending=True)

    # Summary by User
    user_summary_df = squeue_exploded_df.group_by(["user_name", "job_state"]).agg(
        pl.len().alias("count")
    ).pivot(
        index="user_name",
        on="job_state",
        values="count",
        aggregate_function="first"
    ).fill_null(0)

    # Add Total Jobs column for user summary
    total_jobs_user = squeue_exploded_df.group_by("user_name").agg(pl.len().alias("Total Jobs"))
    user_summary_df = user_summary_df.join(total_jobs_user, on="user_name", how="left")

    # Reorder columns to have Total Jobs first for user summary
    user_summary_df = user_summary_df.select([
        "user_name", "Total Jobs", pl.exclude("user_name", "Total Jobs")
    ]).sort("Total Jobs", descending=True)

    return account_summary_df, user_summary_df

def process_node_summary(scontrol_data):
    if not scontrol_data or "nodes" not in scontrol_data:
        print("Failed to retrieve scontrol data for node summary.")
        return None

    nodes_df = pl.DataFrame(scontrol_data["nodes"])

    total_nodes = nodes_df.shape[0]
    total_cpus = nodes_df.select(pl.col("cpus")).sum().item()
    total_memory = nodes_df.select(pl.col("real_memory")).sum().item() # real_memory is in MB

    allocated_cpus = nodes_df.select(pl.col("alloc_cpus")).sum().item()
    allocated_memory = nodes_df.select(pl.col("alloc_memory")).sum().item() # alloc_memory is in MB

    # Count broken nodes
    broken_states = ["DRAINED", "DRAINING", "DRAIN", "DOWN", "FAIL", "NO_RESPOND", "POWER_DOWN", "POWER_UP", "RESUME", "UNKNOWN"]
    broken_nodes_condition = None
    for s in broken_states:
        if broken_nodes_condition is None:
            broken_nodes_condition = pl.col("state").list.contains(s)
        else:
            broken_nodes_condition = broken_nodes_condition | pl.col("state").list.contains(s)
    
    broken_nodes = nodes_df.filter(broken_nodes_condition).shape[0]

    # Count nodes in reservation
    reservation_nodes = nodes_df.filter(
        pl.col("state").list.contains("RESERVATION")
    ).shape[0]

    # Count mixed nodes (some, but not all, CPUs are allocated)
    mixed_nodes = nodes_df.filter(
        (pl.col("alloc_cpus") > 0) & (pl.col("alloc_cpus") < pl.col("cpus"))
    ).shape[0]

    summary_data = {
        "Metric": [
            "Total Nodes", "Total CPUs", "Total Memory (MB)",
            "Allocated CPUs", "Allocated Memory (MB)",
            "Broken Nodes", "Nodes in Reservation", "Mixed Nodes"
        ],
        "Value": [
            total_nodes, total_cpus, total_memory,
            allocated_cpus, allocated_memory,
            broken_nodes, reservation_nodes, mixed_nodes
        ]
    }
    return pl.DataFrame(summary_data)

def format_seconds_to_human_readable(seconds):
    if seconds is None:
        return "N/A"
    seconds = int(seconds)
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{days}d {hours}h {minutes}m {seconds}s"

def process_pending_job_waiting_times_summary(squeue_data):
    if not squeue_data or "jobs" not in squeue_data or not squeue_data["jobs"]:
        print("Failed to retrieve squeue data for pending job waiting times summary or no jobs found.")
        # Return a DataFrame with the expected structure but zero values
        return pl.DataFrame({
            "Metric": ["Max Waiting Time", "Median Waiting Time", "Mean Waiting Time"],
            "Value": ["0d 0h 0m 0s"] * 3
        })

    squeue_df = pl.DataFrame(squeue_data["jobs"])

    pending_jobs_df = squeue_df.filter(
        (pl.col("job_state").list.contains("PENDING")) &
        (pl.col("eligible_time").struct.field("number") != 0) &
        (pl.col("start_time").struct.field("number") != 0)
    ).with_columns([
        (pl.col("start_time").struct.field("number") - pl.col("eligible_time").struct.field("number")).alias("waiting_time_seconds")
    ])

    if pending_jobs_df.is_empty():
        print("No pending jobs found to calculate waiting times.")
        return pl.DataFrame({
            "Metric": ["Max Waiting Time", "Median Waiting Time", "Mean Waiting Time"],
            "Value": ["0d 0h 0m 0s"] * 3
        })

    max_waiting_time = float(pending_jobs_df.select(pl.col("waiting_time_seconds").max()).item() or 0.0)
    median_waiting_time = float(pending_jobs_df.select(pl.col("waiting_time_seconds").median()).item() or 0.0)
    mean_waiting_time = float(pending_jobs_df.select(pl.col("waiting_time_seconds").mean()).item() or 0.0)

    summary_df = pl.DataFrame({
        "Metric": ["Max Waiting Time", "Median Waiting Time", "Mean Waiting Time"],
        "Value": [max_waiting_time, median_waiting_time, mean_waiting_time]
    })

    summary_df = summary_df.with_columns(
        pl.col("Value").map_elements(format_seconds_to_human_readable, return_dtype=pl.String).alias("Value")
    )

    return summary_df

async def get_all_slurm_data():
    """Fetches all Slurm data concurrently."""
    sinfo_task = get_slurm_data("sinfo --json")
    squeue_task = get_slurm_data("squeue --json")
    scontrol_task = get_slurm_data("scontrol show nodes --json")
    
    results = await asyncio.gather(sinfo_task, squeue_task, scontrol_task)
    
    return {
        "sinfo": results[0],
        "squeue": results[1],
        "scontrol": results[2]
    }

from textual.screen import Screen
from textual.widgets import Static
from textual.binding import Binding

class AboutScreen(Screen):
    BINDINGS = [
        ("b", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("""# About This Application

This application provides a real-time monitoring dashboard for a Slurm cluster. It fetches data from Slurm using `sinfo`, `squeue`, and `scontrol` commands and presents it in an easy-to-read tabular format.

## Features:
- **Partition Summary:** Overview of nodes and CPUs per partition (total, idle, allocated).
- **Node Summary:** Detailed statistics on total nodes, CPUs, memory, and counts of broken, reserved, and mixed nodes.
- **Job Summaries:** Aggregated views of running, pending, and completed jobs by account and user.
- **Pending Job Waiting Times:** Provides insights into the maximum, median, and mean waiting times for pending jobs.

The data is refreshed every 30 seconds to provide up-to-date information on your Slurm cluster's status.
""", classes="about-text")
        )
        yield Footer()

class HelpScreen(Screen):
    BINDINGS = [
        ("b", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("""# Help

This application displays various metrics from a Slurm cluster.

## Key Bindings:
- `d`: Toggle dark mode
- `q`: Quit the application
- `a`: Show About page
- `h`: Show this Help page
- `b`: Go back to the main screen

## Data Tables:
- **Partition Summary:** Shows resource allocation per Slurm partition.
- **Node Summary:** Provides an overview of node health and resource utilization.
- **Job Summary:** Displays total running and pending jobs.
- **User Job Summary:** Breaks down job counts by user.
- **Account Job Summary:** Breaks down job counts by account.

## Troubleshooting:
- If data tables are empty, ensure Slurm commands (`sinfo`, `squeue`, `scontrol`) are installed and accessible in your system's PATH.
- Check for error messages in the console output for more details on data retrieval issues.
""", classes="help-text")
        )
        yield Footer()

async def get_slurm_text_data(command):
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            print(f"Error running {command}: {stderr.decode().strip()}")
            return None
        return stdout.decode().strip()
    except FileNotFoundError:
        print(f"Error: Command '{command.split()[0]}' not found. Is Slurm installed and in your PATH?")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while running {command}: {e}")
        return None

async def fetch_config():
    """Fetches Slurm configuration."""
    return await get_slurm_text_data("scontrol show config")

class ConfigScreen(Screen):
    BINDINGS = [
        ("b", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer():
            yield Static("", id="config_static")
        yield Footer()

    async def on_mount(self) -> None:
        config_data = await fetch_config()
        static = self.query_one("#config_static", Static)
        if config_data:
            static.update(config_data)
        else:
            static.update("Failed to fetch Slurm configuration.")


class CustomFooter(Footer):
    """A custom footer that includes a last updated timestamp."""

    def compose(self) -> ComposeResult:
        yield Static("", id="last_updated")

    def update_timestamp(self) -> None:
        self.query_one("#last_updated", Static).update(
            f"Last updated: {datetime.datetime.now().ctime()}"
        )


import argparse

import datetime


class SlurmMonitorApp(App):
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
        ("a", "push_screen('about')", "About"),
        ("h", "push_screen('help')", "Help"),
        ("c", "push_screen('config')", "Config"),
    ]

    SCREENS = {
        "about": AboutScreen,
        "help": HelpScreen,
        "config": ConfigScreen,
    }

    def __init__(self, delay: int):
        super().__init__()
        self.delay = delay

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield DataTable(id="partition_summary_table")
            yield DataTable(id="node_summary_table")
            yield DataTable(id="job_summary_table")
        with Horizontal():
            with Vertical():
                yield DataTable(id="user_job_summary_table")
            with Vertical():
                yield DataTable(id="account_job_summary_table")
        yield CustomFooter()

    async def on_mount(self) -> None:
        self.set_interval(self.delay, self.update_data)

        # Initialize table columns once
        all_data = await get_all_slurm_data()
        sinfo_data = all_data["sinfo"]
        squeue_data = all_data["squeue"]
        scontrol_data = all_data["scontrol"]

        partition_table = self.query_one("#partition_summary_table", DataTable)
        partition_summary = process_partition_summary(sinfo_data)
        if partition_summary is not None:
            partition_table.add_columns(*partition_summary.columns)

        node_table = self.query_one("#node_summary_table", DataTable)
        node_summary = process_node_summary(scontrol_data)
        if node_summary is not None:
            node_table.add_columns(*node_summary.columns)

        job_table = self.query_one("#job_summary_table", DataTable)
        job_summary_initial_df = pl.DataFrame({
            "Metric": ["Running Jobs", "Pending Jobs"],
            "Value": [0, 0]
        })
        job_table.add_columns(*job_summary_initial_df.columns)

        user_job_table = self.query_one("#user_job_summary_table", DataTable)
        account_job_table = self.query_one("#account_job_summary_table", DataTable)
        
        # Get initial job summaries to set columns
        account_summary, user_summary = process_job_summaries(squeue_data)
        if account_summary is not None:
            account_job_table.add_columns(*["account", "Total Jobs", "RUNNING", "PENDING", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "SUSPENDED"])
        if user_summary is not None:
            user_job_table.add_columns(*["user_name", "Total Jobs", "RUNNING", "PENDING", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "SUSPENDED"])

        await self.update_data()

    async def update_data(self) -> None:
        all_data = await get_all_slurm_data()
        sinfo_data = all_data["sinfo"]
        squeue_data = all_data["squeue"]
        scontrol_data = all_data["scontrol"]

        # Update Partition Summary
        partition_summary = process_partition_summary(sinfo_data)
        if partition_summary is not None:
            partition_table = self.query_one("#partition_summary_table", DataTable)
            partition_table.clear()
            partition_table.add_rows(partition_summary.rows())

        # Update Node Summary
        node_summary = process_node_summary(scontrol_data)
        if node_summary is not None:
            node_table = self.query_one("#node_summary_table", DataTable)
            node_table.clear()
            node_table.add_rows(node_summary.rows())

        # Update Job Summary
        account_summary, user_summary = process_job_summaries(squeue_data)
        if account_summary is not None and user_summary is not None:
            total_running_jobs = account_summary.select(pl.sum("RUNNING")).item() if "RUNNING" in account_summary.columns else 0
            total_pending_jobs = account_summary.select(pl.sum("PENDING")).item() if "PENDING" in account_summary.columns else 0
            
            job_summary_df = pl.DataFrame({
                "Metric": ["Running Jobs", "Pending Jobs"],
                "Value": [total_running_jobs, total_pending_jobs]
            })

            job_table = self.query_one("#job_summary_table", DataTable)
            job_table.clear()
            job_table.add_rows(job_summary_df.rows())

            # Update User Job Summary
            user_job_table = self.query_one("#user_job_summary_table", DataTable)
            user_job_table.clear()
            user_job_table.add_rows(user_summary.rows())

            # Update Account Job Summary
            account_job_table = self.query_one("#account_job_summary_table", DataTable)
            account_job_table.clear()
            account_job_table.add_rows(account_summary.rows())

        self.query_one(CustomFooter).update_timestamp()

def main():
    parser = argparse.ArgumentParser(
        description="A real-time monitoring dashboard for a Slurm cluster.",
        epilog="""
Key Bindings:
  d: Toggle dark mode
  q: Quit the application
  a: Show About page
  h: Show Help page
  c: Show Slurm config page
  b: Go back to the main screen
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--delay", type=int, default=30, help="The delay between updates in seconds.")
    args = parser.parse_args()

    app = SlurmMonitorApp(delay=args.delay)
    app.run()

if __name__ == '__main__':
    main()

