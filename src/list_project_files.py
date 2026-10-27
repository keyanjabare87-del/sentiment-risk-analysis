
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for folder_name in ["data", "outputs"]:
    folder = ROOT / folder_name

    print(f"\n{'=' * 55}")
    print(f"目录：{folder}")

    if not folder.exists():
        print("目录不存在")
        continue

    for path in sorted(folder.rglob("*")):
        if path.is_file():
            print(path.relative_to(ROOT))
