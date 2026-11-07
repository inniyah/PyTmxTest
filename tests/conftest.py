import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # helpers de los tests

MAP01 = ROOT / "maps" / "map01" / "VTiles1.tmx"
MAP02 = ROOT / "maps" / "map02" / "map02.tmx"
