import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # helpers de los tests

MAP01 = ROOT / "maps" / "map01" / "VTiles1.tmx"
MAP02 = ROOT / "maps" / "map02" / "map02.tmx"


import pytest  # noqa: E402


@pytest.fixture(scope="session")
def shared_tileset(tmp_path_factory):
    """El tileset de prueba, generado una sola vez por sesión (tarda unos segundos)."""
    from tools.pieces.build import build_tileset
    return build_tileset(tmp_path_factory.mktemp("shared") / "assets")


def retrying_viewer(viewer_cls, attempts: int = 4, pause: float = 0.5):
    """El servidor gráfico virtual de los tests (xvfb) a veces no deja iniciar GLFW cuando se
    abren varias ventanas seguidas: se reintenta un par de veces antes de dar el test por fallado.
    Solo para los tests; el motor no cambia."""
    import time

    def make(*args, **kwargs):
        for i in range(attempts):
            try:
                return viewer_cls(*args, **kwargs)
            except RuntimeError as e:
                if "GLFW" not in str(e) or i == attempts - 1:
                    raise
                time.sleep(pause)
    return make
