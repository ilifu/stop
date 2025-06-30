import datetime
import json
from importlib import resources
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Input, Static
from textual.containers import Container, ScrollableContainer
from textual.screen import Screen
from textual.binding import Binding
import polars as pl

from .slurm import (
    get_all_slurm_data,
    get_slurm_data,
    process_partition_summary,
    process_node_summary,
    process_job_summaries,
    process_node_list,
    process_partition_list,
    fetch_config,
    process_pending_job_wait_time_stats,
)

class AboutScreen(Screen):
    BINDINGS = [
        ("b", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield ScrollableContainer(
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
        yield ScrollableContainer(
            Static("""# Help

This application displays various metrics from a Slurm cluster.

## Key Bindings:
- `d`: Toggle dark mode
- `q`: Quit the application
- `a`: Show About page
- `h`: Show this Help page
- `n`: Show Node list page
- `p`: Show Partition list page
- `Enter`: Show details for selected item
- `/`: Search in lists
- `ESC`: Exit search mode
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

class NodeScreen(Screen):
    BINDINGS = [
        ("b", "app.pop_screen", "Back"),
        ("r", "refresh_nodes", "Refresh"),
        ("/", "show_search", "Search"),
        ("escape", "hide_search", "Hide Search"),
    ]

    def __init__(self, delay: int):
        super().__init__()
        self.delay = delay
        self.node_list = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Search nodes...", id="search_input")
        yield DataTable(id="node_list_table", cursor_type="row")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one(DataTable).focus()
        self.set_interval(self.delay, self.update_nodes)
        await self.update_nodes()
        self.query_one("#search_input").display = False

    async def update_nodes(self) -> None:
        scontrol_data = await get_slurm_data("scontrol show nodes --json")
        self.node_list = process_node_list(scontrol_data)
        self.filter_nodes(self.query_one("#search_input").value)

    def filter_nodes(self, search_term: str = "") -> None:
        table = self.query_one("#node_list_table", DataTable)
        if self.node_list is not None:
            if not table.columns:
                table.add_columns(*self.node_list.columns)
            
            filtered_list = self.node_list
            if search_term:
                filtered_list = self.node_list.filter(
                    pl.col("Node Name").str.contains(search_term, literal=True)
                )
            
            table.clear()
            table.add_rows(filtered_list.rows())

    async def action_refresh_nodes(self) -> None:
        await self.update_nodes()

    def action_show_search(self) -> None:
        search_input = self.query_one("#search_input")
        search_input.display = True
        search_input.focus()

    def action_hide_search(self) -> None:
        search_input = self.query_one("#search_input")
        if search_input.display:
            search_input.value = ""
            search_input.display = False
            self.query_one(DataTable).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self.filter_nodes(event.value)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = self.query_one(DataTable)
        node_name = table.get_row_at(event.cursor_row)[0]
        self.app.push_screen(NodeDetailScreen(node_name=node_name))

class NodeDetailScreen(Screen):
    BINDINGS = [
        ("b", "app.pop_screen", "Back"),
    ]

    def __init__(self, node_name: str):
        super().__init__()
        self.node_name = node_name

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer():
            yield Static(id="node_details")
        yield Footer()

    async def on_mount(self) -> None:
        await self.update_node_details()

    async def update_node_details(self) -> None:
        node_data = await get_slurm_data(f"scontrol show node {self.node_name} --json")
        static = self.query_one("#node_details", Static)
        if node_data:
            static.update(json.dumps(node_data, indent=4))
        else:
            static.update(f"Failed to fetch details for node {self.node_name}.")

class PartitionScreen(Screen):
    BINDINGS = [
        ("b", "app.pop_screen", "Back"),
        ("r", "refresh_partitions", "Refresh"),
        ("/", "show_search", "Search"),
        ("escape", "hide_search", "Hide Search"),
    ]

    def __init__(self, delay: int):
        super().__init__()
        self.delay = delay
        self.partition_list = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Search partitions...", id="search_input")
        yield DataTable(id="partition_list_table", cursor_type="row")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one(DataTable).focus()
        self.set_interval(self.delay, self.update_partitions)
        await self.update_partitions()
        self.query_one("#search_input").display = False

    async def update_partitions(self) -> None:
        sinfo_data = await get_slurm_data("sinfo --json")
        self.partition_list = process_partition_list(sinfo_data)
        self.filter_partitions(self.query_one("#search_input").value)

    def filter_partitions(self, search_term: str = "") -> None:
        table = self.query_one("#partition_list_table", DataTable)
        if self.partition_list is not None:
            if not table.columns:
                table.add_columns(*self.partition_list.columns)
            
            filtered_list = self.partition_list
            if search_term:
                filtered_list = self.partition_list.filter(
                    pl.col("Partition Name").str.contains(search_term, literal=True)
                )
            
            table.clear()
            table.add_rows(filtered_list.rows())

    async def action_refresh_partitions(self) -> None:
        await self.update_partitions()

    def action_show_search(self) -> None:
        search_input = self.query_one("#search_input")
        search_input.display = True
        search_input.focus()

    def action_hide_search(self) -> None:
        search_input = self.query_one("#search_input")
        if search_input.display:
            search_input.value = ""
            search_input.display = False
            self.query_one(DataTable).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self.filter_partitions(event.value)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = self.query_one(DataTable)
        partition_name = table.get_row_at(event.cursor_row)[0]
        self.app.push_screen(PartitionDetailScreen(partition_name=partition_name))

class PartitionDetailScreen(Screen):
    BINDINGS = [
        ("b", "app.pop_screen", "Back"),
    ]

    def __init__(self, partition_name: str):
        super().__init__()
        self.partition_name = partition_name

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer():
            yield Static(id="partition_details")
        yield Footer()

    async def on_mount(self) -> None:
        await self.update_partition_details()

    async def update_partition_details(self) -> None:
        partition_data = await get_slurm_data(f"scontrol show partition {self.partition_name} --json")
        static = self.query_one("#partition_details", Static)
        if partition_data:
            static.update(json.dumps(partition_data, indent=4))
        else:
            static.update(f"Failed to fetch details for partition {self.partition_name}.")

class CustomFooter(Footer):
    """A custom footer that includes a last updated timestamp."""

    def compose(self) -> ComposeResult:
        yield Static("", id="last_updated")

    def update_timestamp(self) -> None:
        self.query_one("#last_updated", Static).update(
            f"Last updated: {datetime.datetime.now().ctime()}"
        )

class SlurmMonitorApp(App):
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
        ("a", "push_screen('about')", "About"),
        ("h", "push_screen('help')", "Help"),
        ("c", "push_screen('config')", "Config"),
        ("n", "push_screen('nodes')", "Nodes"),
        ("p", "push_screen('partitions')", "Partitions"),
    ]

    SCREENS = {
        "about": AboutScreen,
        "help": HelpScreen,
        "config": ConfigScreen,
    }
    
    def __init__(self, delay: int):
        super().__init__()
        self.delay = delay

    def on_load(self) -> None:
        with resources.path(__package__, "tui.css") as p:
            self.stylesheet_path = p

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield DataTable(id="partition_summary_table")
            yield DataTable(id="node_summary_table")
            yield DataTable(id="job_summary_table")
            yield DataTable(id="pending_job_wait_time_stats_table")
            yield DataTable(id="user_job_summary_table")
            yield DataTable(id="account_job_summary_table")
        yield CustomFooter()

    async def on_mount(self) -> None:
        self.install_screen(NodeScreen(delay=self.delay), "nodes")
        self.install_screen(PartitionScreen(delay=self.delay), "partitions")
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
            "Job Summary": ["Running Jobs", "Pending Jobs"],
            "Value": [0, 0]
        })
        job_table.add_columns(*job_summary_initial_df.columns)

        pending_job_wait_time_stats_table = self.query_one("#pending_job_wait_time_stats_table", DataTable)
        pending_job_wait_time_stats_summary = process_pending_job_wait_time_stats(squeue_data, all_data["current_time"])
        if pending_job_wait_time_stats_summary is not None:
            pending_job_wait_time_stats_table.add_columns(*pending_job_wait_time_stats_summary.columns)

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
                "Job Summary": ["Running Jobs", "Pending Jobs"],
                "Value": [total_running_jobs, total_pending_jobs]
            })

            job_table = self.query_one("#job_summary_table", DataTable)
            job_table.clear()
            job_table.add_rows(job_summary_df.rows())

            # Update Pending Job Wait Time Stats
            pending_job_wait_time_stats_summary = process_pending_job_wait_time_stats(squeue_data, all_data["current_time"])
            if pending_job_wait_time_stats_summary is not None:
                pending_job_wait_time_stats_table = self.query_one("#pending_job_wait_time_stats_table", DataTable)
                pending_job_wait_time_stats_table.clear()
                pending_job_wait_time_stats_table.add_rows(pending_job_wait_time_stats_summary.rows())

            # Update User Job Summary
            user_job_table = self.query_one("#user_job_summary_table", DataTable)
            user_job_table.clear()
            user_job_table.add_rows(user_summary.rows())

            # Update Account Job Summary
            account_job_table = self.query_one("#account_job_summary_table", DataTable)
            account_job_table.clear()
            account_job_table.add_rows(account_summary.rows())

        self.query_one(CustomFooter).update_timestamp()

