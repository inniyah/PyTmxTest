"""Motor nuevo: mundo compilado de celdas de 32 bits, sin dependencia de Tiled en tiempo de juego.

    engine.world      formato de celda, almacenamiento (WorldStore) y mundo cargado en memoria
    engine.compiler   Tiled (.tmx) -> mundo compilado (solo herramienta, nunca en el juego)
"""

# PyOpenGL comprueba si hubo un error después de cada llamada a OpenGL: útil para depurar y caro
# en un bucle de dibujo. Se desactiva aquí, antes de que ningún módulo importe OpenGL.GL; con
# la variable de entorno ENGINE_GL_DEBUG=1 se mantiene activo.
import os as _os

if _os.environ.get("ENGINE_GL_DEBUG") != "1":
    try:
        import OpenGL as _OpenGL
        _OpenGL.ERROR_CHECKING = False
    except ImportError:                    # herramientas sin OpenGL (compilador, tests sin GPU)
        pass
