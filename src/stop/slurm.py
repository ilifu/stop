import asyncio
import json
import polars as pl

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

async def fetch_config():
    """Fetches Slurm configuration."""
    return await get_slurm_text_data("scontrol show config")

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

def process_node_list(scontrol_data):
    if not scontrol_data or "nodes" not in scontrol_data:
        print("Failed to retrieve scontrol data for node list.")
        return None

    nodes_df = pl.DataFrame(scontrol_data["nodes"])

    # Select and rename columns
    node_list_df = nodes_df.select([
        pl.col("name").alias("Node Name"),
        pl.col("state").list.join(", ").alias("State"),
        pl.col("cpus").alias("Total Cores"),
        pl.col("alloc_cpus").alias("Allocated Cores"),
        pl.col("real_memory").alias("Total Memory (MB)"),
        pl.col("alloc_memory").alias("Allocated Memory (MB)"),
        (pl.col("real_memory") - pl.col("alloc_memory")).alias("Free Memory (MB)"),
        pl.col("partitions").list.join(", ").alias("Partitions"),
        pl.col("cpu_load").alias("CPU Load"),
    ])

    return node_list_df.sort("Node Name")

def process_partition_list(sinfo_data):
    if not sinfo_data or "sinfo" not in sinfo_data:
        print("Failed to retrieve sinfo data for partition list.")
        return None

    partitions_df = pl.DataFrame(sinfo_data["sinfo"])

    # Define aggregations
    aggs = [
        pl.col("nodes").struct.field("total").sum().alias("Total Nodes"),
        pl.col("nodes").struct.field("idle").sum().alias("Idle Nodes"),
        pl.col("nodes").struct.field("allocated").sum().alias("Allocated Nodes"),
        pl.col("cpus").struct.field("total").sum().alias("Total CPUs"),
        pl.col("cpus").struct.field("idle").sum().alias("Free CPUs"),
        pl.col("cpus").struct.field("allocated").sum().alias("Allocated CPUs"),
    ]

    if "availability" in partitions_df.columns:
        aggs.append(pl.col("availability").unique().list().join(", ").alias("Availability"))
    
    if "state" in partitions_df.columns:
        aggs.append(pl.col("state").unique().list().join(", ").alias("State"))

    # Group by partition name and aggregate
    partition_list_df = partitions_df.group_by(
        pl.col("partition").struct.field("name").alias("Partition Name")
    ).agg(aggs)

    return partition_list_df.sort("Partition Name")

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
