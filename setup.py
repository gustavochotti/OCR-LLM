import subprocess
import sys

def install():
  try:
    print(f"Installing: {command}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + comando.split())
    # print(f"Succesfully installed: {comando}\n")
  except subprocess.CalledProcessError as e:
    print(f"Failed to execute command: {e}")
    sys.exit(1)

c_torch = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
install(c_torch)

c_requirements = "-r requirements.txt"
install(c_requirements)