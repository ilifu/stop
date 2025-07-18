#!/usr/bin/env python3
import argparse
from tui import SlurmMonitorApp

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
  n: Show node list page
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