import pyglet
from pyglet.graphics import Batch
from pyglet.gl import *
from .animations import *
from pybiomech.viz.camera import Camera

class Scene(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opaque_batch = Batch()  # For objects with alpha = 1.0
        self.transparent_batch = Batch()  # For objects with alpha < 1.0
        self.animations = []  # Queue of animations to play
        self.is_playing = False
        self.record = kwargs.get("record", False)  # Option to record animations
        self.camera = None
        self.objects = []
    
    def add(self, obj):
        self.objects.append(obj)

    def on_draw(self):
        """
        Renders the scene.
        """
        self.clear()
        glClearColor(0.1, 0.1, 0.1, 1.0)  # Background color
        
        # draw in the objects that are not in the batch already
        for obj in self.objects:
            obj.draw()

        # draw opaque objects
        self.opaque_batch.draw()
        
        # now render the transparent objects
        glEnable(GL_BLEND)
        self.transparent_batch.draw()
        glDisable(GL_BLEND)
    
    def addCamera(self, *args, **kwargs):
        self.camera = Camera(*args, **kwargs)
    
    def construct(self):
        pass # implemented by subclass
    
    def render(self):
        pyglet.clock.schedule(self.construct, 0.0)
        pyglet.app.run()
    
    def update(self, dt):
        pass
    
    def play3d(self):
        pyglet.clock.schedule_interval(self.update, 1/120.0)
        pyglet.app.run()
        

