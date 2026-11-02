"""
TreadVision Launcher Script
Launches the Streamlit application with appropriate configuration.
"""
import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "app" / "main.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",
        "--theme.base=dark"
    ]
    print(f"Launching TreadVision: {' '.join(cmd)}")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTreadVision terminated.")

if __name__ == "__main__":
    main()
