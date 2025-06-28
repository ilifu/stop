# stop - A Slurm Monitoring Dashboard

`stop` is a terminal-based dashboard for real-time monitoring of a Slurm cluster, inspired by `top`. It provides a quick and easy-to-read overview of your cluster's status, including partition load, node health, and job queues.

The application is built with Python using the [Textual](https://github.com/Textualize/textual) framework.

## Features

- **Real-time Data:** Fetches and displays data from `sinfo`, `squeue`, and `scontrol` at a configurable interval.
- **Comprehensive Views:**
    - **Partition Summary:** Overview of nodes and CPUs per partition (total, idle, allocated).
    - **Node Summary:** High-level statistics on nodes, CPUs, and memory.
    - **Job Summaries:** Aggregated views of jobs by account and user.
    - **Pending Job Stats:** Insights into max, median, and mean waiting times for pending jobs.
- **Interactive Screens:**
    - An **About** page with application details.
    - A **Help** page explaining key bindings and data tables.
    - A **Config** page displaying the output of `scontrol show config`.
- **Customization:**
    - Configurable data refresh delay via the `--delay` flag.
    - Dark and light mode support.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd stop
    ```

2.  **Install the application:**
    Using `pip`, you can install the application and its dependencies. This command also installs the `stop` executable and the associated man page.

    ```bash
    pip install .
    ```

    It is highly recommended to do this within a Python virtual environment.

## Usage

Once installed, you can run the application by simply typing:

```bash
stop
```

### Command-line Options

-   `--delay SECONDS`: Set the data refresh interval in seconds (default: 30).
    ```bash
    stop --delay 10
    ```
-   `--help`: Show the help message, including a list of key bindings.

### Man Page

A man page is included with the installation. View it with:
```bash
man stop
```

### Key Bindings

The application can be controlled with the following keys:

| Key | Action                                       |
|-----|----------------------------------------------|
| `d` | Toggle dark/light mode                       |
| `q` | Quit the application                         |
| `a` | Show the "About" screen                      |
| `h` | Show the "Help" screen                       |
| `c` | Show the "Slurm Config" screen               |
| `n` | Show the "Node List" screen                  |
| `p` | Show the "Partition List" screen             |
| `Enter` | Show details for selected item in a list     |
| `b` | Go back to the main screen (from other screens) |


## Development

To set up a development environment, clone the repository and install the project in editable mode with its development dependencies:

```bash
# Clone the repo (if you haven't already)
git clone <repository-url>
cd stop

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e '.[dev]'
```

### Running Tests

The project uses `pytest`. To run the test suite:

```bash
uv run pytest
```
