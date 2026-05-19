"""检查 Ollama 是否可用"""
import subprocess
result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
if result.returncode == 0:
    print("Ollama 已安装")
    print(result.stdout)
else:
    print("Ollama 不可用")
    print(result.stderr)
