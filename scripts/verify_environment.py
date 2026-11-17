"""核对实际解释器与项目锁定版本。运行 python scripts/verify_environment.py。"""

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main() -> int:
    baseline = json.loads(
        (Path(__file__).resolve().parents[1] / "release-baseline.json").read_text(
            encoding="utf-8"
        )
    )
    errors = []
    print(f"Python: {sys.version.split()[0]}\n解释器: {sys.executable}")
    minimum = tuple(int(part) for part in baseline["python_minimum"].split("."))
    if sys.version_info[:2] < minimum:
        errors.append("LangChain 1.5 / Core 1.8 需要 Python >=3.11")
    for name, expected in baseline["packages"].items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = "未安装"
        print(f"{name}: {actual}（预期 {expected}）")
        if actual != expected:
            errors.append(f"{name} 版本不一致")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
