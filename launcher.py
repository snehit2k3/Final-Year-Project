import subprocess
import sys
import os

# Determine the path to the batch file (PyInstaller handles this pathing)
# For --onefile, the base directory is accessible via sys._MEIPASS
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_dir = sys._MEIPASS
else:
    # Running as a regular script (for testing)
    base_dir = os.path.dirname(os.path.abspath(__file__))

# The path to your batch script relative to the base directory
batch_path = os.path.join(base_dir, 'startup.bat')

print(f"Launching batch script: {batch_path}")

try:
    # Execute the batch file using the system's shell
    # This command must be executed using the shell=True option for .bat files
    subprocess.run([batch_path], shell=True, check=True)
except Exception as e:
    print(f"An error occurred while running the script: {e}")

# Keep the console window open briefly if not using --noconsole,
# but since you are using --noconsole, this script exits,
# leaving the servers (launched by `start cmd /k`) running.