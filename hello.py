from pybiomech.viz import Scene
import pyglet
from pybiomech.physics import Frame


def main():
    scene = Scene(width = 1280, height = 720, caption = "Tesitng this out!")

    label = pyglet.text.Label(
        "Hello World",
        font_name='Times New Roman',
        font_size=36,
        x=scene.width//2, y=scene.height//2,
        anchor_x='center', anchor_y='center',
        color = [255, 255, 255, 255],
        batch = scene.opaque_batch,
    )

    box = pyglet.shapes.Box(
        x=scene.width//2, y=scene.height//2,
        width = 100.0, height = 100.0,
        thickness = 10.0,
        color = [255, 255, 255, 80],
        batch = scene.transparent_batch,
    )

    pyglet.app.run()
    


if __name__ == "__main__":
    main()
