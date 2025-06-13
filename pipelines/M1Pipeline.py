import subprocess
from datetime import datetime
from pathlib import Path

# Base path del progetto
base_path = Path(__file__).resolve().parents[1] / "src" / "SVM"

# Percorsi manuali degli script da eseguire
scripts = [
    base_path / "3_PowerOfNKernel.py",
    base_path / "5_FibonacciIndexKernel.py",
    base_path / "6_ZigZagKernel.py",
    base_path / "8_DivisibleIndexKernel.py",
    base_path / "11_ReverseTunnelWindowKernel.py"
]

# Crea cartella log con timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_path = Path(__file__).parent
log_dir = file_path / timestamp
log_dir.mkdir(exist_ok=True, parents=True)

# Esegui ogni script e salva l'output
for script in scripts:
    script_path = Path(script)
    log_file = log_dir / f"{script_path.stem}.log"
    with open(log_file, "w") as out:
        print(f"Running: {script_path}")
        subprocess.run(
            ["python", str(script_path)],
            stdout=out,
            stderr=subprocess.STDOUT  # cattura anche errori
        )