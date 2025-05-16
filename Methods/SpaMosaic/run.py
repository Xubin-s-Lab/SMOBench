
import subprocess

# 定义 Python 脚本的路径


# python_script = "/home/users/nus/dmeng/scratch/spbench/swruan/SpaMosaic-main/Human_Lymph_Node/HLN.py"

# python_script = "/home/users/nus/dmeng/scratch/spbench/swruan/SpaMosaic-main/Human_Lymph_Node/HT.py"

# python_script = "/home/users/nus/dmeng/scratch/spbench/swruan/SpaMosaic-main/Human_Lymph_Node/MISARs2.py"


# python_script = "/home/users/nus/dmeng/scratch/spbench/swruan/SpaMosaic-main/Human_Lymph_Node/MISAR.py"

# python_script = "/home/users/nus/dmeng/scratch/spbench/swruan/SpaMosaic-main/Human_Lymph_Node/mouse_Spleen.py"

# python_script = "/home/users/nus/dmeng/scratch/spbench/swruan/SpaMosaic-main/Human_Lymph_Node/Mouse_Thymus.py"

python_script = "/home/users/nus/dmeng/scratch/spbench/swruan/SpaMosaic-main/Human_Lymph_Node/Mouse_Brain.py"

venv_python = "/home/users/nus/dmeng/miniconda3/envs/SpaMosaic/bin/python"

# 构造命令
command = [
    venv_python, python_script,
]

# 运行命令
try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Python script executed successfully.")
except subprocess.CalledProcessError as e:
    print("Python script failed with return code:", e.returncode)
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)