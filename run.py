import subprocess
import sys
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd=project_dir
)
print("Process started:", proc.pid)
