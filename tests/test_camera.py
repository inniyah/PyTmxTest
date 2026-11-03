from tmx_explorer.camera import Camera


def test_fit_lowers_min_zoom_so_wheel_never_jumps():
    cam = Camera(1280, 720)
    cam.fit(0, 0, 3840, 60000)                     # mapa muy alto: zoom muy pequeño
    fitted = cam.zoom
    assert fitted < Camera.MIN_ZOOM
    cam.zoom_by(1.1)
    assert abs(cam.zoom - fitted * 1.1) < 1e-9     # sin salto a MIN_ZOOM
    cam.zoom_by(1 / 1.1)
    cam.zoom_by(1 / 1.1)
    assert abs(cam.zoom - fitted) < 1e-9           # y se puede volver a ver todo


def test_fit_centers_the_rectangle():
    cam = Camera(800, 600)
    cam.fit(100, 200, 400, 300)
    assert cam.center == (300.0, 350.0)
