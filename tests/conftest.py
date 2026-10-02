"""pytest 공통 설정."""

import pathlib
import sys

# src 레이아웃을 설치 없이도 import 할 수 있게 한다.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
