# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 03:19:01 2016

@author: Jeff
"""



from version1.utilities import read_obj
from version1.gfx import *
import version1.gfx.gui as gui
import version1.biomech as bm
import numpy as np
import matplotlib.pyplot as plt







'''
def read_mesh(filename):
    v, f, n = read_obj(filename)
    v = np.array(v) / 1000.0
    f = np.array(f) - 1
    n = np.array(n)
    return Mesh(vertices = v, faces = f, normals = n)


def spin_head(dt):
    mesh_list[0].euler_angle[2] += dt * 100.0
    #mesh_list[0].rotate_mesh_around_point(dt, 0.0, 0.0, np.array([0.0, 0.0, 0.0]))


list_of_files = ["skull", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "Thorax"]
directory = "../obj-files/"
mesh_list = []

for file in list_of_files:
    mesh_list.append(read_mesh(directory + file + ".obj"))

C7_centroid = mesh_list[7].centroid()

for mesh in mesh_list:
    mesh.translate_mesh(-C7_centroid)


window = gui.GFXWindow()
for mesh in mesh_list:
    window.add_object(mesh)
window.add_update_function(spin_head)
window.run()
'''



'''

C1 = read_mesh("../obj-files/C1.obj")
C2 = read_mesh("../obj-files/C2.obj")
C3 = read_mesh("../obj-files/C3.obj")
C4 = read_mesh("../obj-files/C4.obj")
C5 = read_mesh("../obj-files/C5.obj")
C6 = read_mesh("../obj-files/C6.obj")
C7 = read_mesh("../obj-files/C7.obj")

C7_centroid = C7.centroid()
C1.translate(-C7_centroid)
C2.translate(-C7_centroid)
C3.translate(-C7_centroid)
C4.translate(-C7_centroid)
C5.translate(-C7_centroid)
C6.translate(-C7_centroid)
C7.translate(-C7_centroid)

C1_v = np.array(C1_v) / 1000.0
C1_v = C1_v - np.mean(C1_v, axis = 0)
C1_f = np.array(C1_f) - 1
C1_n = np.array(C1_n)

print(C1_n)

C1 = Mesh(vertices = C1_v, faces = C1_f, normals = C1_n)

print(np.array(faces))
vertices = (np.array(vertices) - np.mean(vertices, axis = 0)) / 1000.0
faces = np.array(faces) - 1
normals = np.array(normals)
#faces = np.arange(0,len(faces))
print(faces.flatten())


atlas = Mesh(vertices = vertices, normals = normals, faces = faces)

window = gui.GFXWindow()
window.add_object(atlas)
window.run()
'''




'''
from version1.utilities import read_obj
import pyglet
from pyglet.gl import *
from math import pi, sin, cos
from time import sleep, time
from os import _exit
import ctypes


vertices, normals, faces = read_obj("../obj-files/earth.obj", combine = list.extend)




def hextoint(i):
    if i > 255:
        i = 255
    return (1.0/255.0) * i



def setup():
    # One-time GL setup
    #glClearColor(1, 1, 1, 1)
    #glColor3f(hextoint(0), hextoint(0), hextoint(0))
    #glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    # Uncomment this line for a wireframe view
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    # Simple light setup.  On Windows GL_LIGHT0 is enabled by default,
    # but this is not the case on Linux or Mac, so remember to always
    # include it.
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)

    # Define a simple function to create ctypes arrays of floats:
    def vec(*args):
        return (GLfloat * len(args))(*args)

    glLightfv(GL_LIGHT0, GL_POSITION, vec(.5, .5, 1, 0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, vec(.5, .5, 1, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1, 1, 1, 1))
    glLightfv(GL_LIGHT1, GL_POSITION, vec(1, 0, .5, 0))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, vec(.5, .5, .5, 1))
    glLightfv(GL_LIGHT1, GL_SPECULAR, vec(1, 1, 1, 1))

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(hextoint(30), hextoint(30), hextoint(30), hextoint(255)))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 1))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 10)

    glTranslatef(0, 0, -8)



class TriMesh(object):


    def __init__(self, vertices = [], normals = [], faces = []):
        """
        Purpose: Initializes the mesh object with vertices, normals and faces
        """
        vertices = (GLfloat * len(vertices))(*vertices)
        normals = (GLfloat * len(normals))(*normals)
        faces = (GLuint * len(faces))(*faces)

        # Compile a display list
        self.list = glGenLists(1)
        glNewList(self.list, GL_COMPILE)

        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glNormalPointer(GL_FLOAT, 0, normals)
        glDrawElements(GL_TRIANGLES, len(faces), GL_UNSIGNED_INT, faces)
        glPopClientAttrib()

        glEndList()

    def draw(self):
        glCallList(self.list)



class gui (pyglet.window.Window):
    def __init__ (self, width=1200, height=900):
        super(gui, self).__init__(650, 450, vsync=True, fullscreen = False, resizable=True, config=Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True,))

        self.colorscheme = {
            'background' : (hextoint(0), hextoint(0), hextoint(0), hextoint(255))
        }
        glClearColor(*self.colorscheme['background'])
        self.set_location(650,300)
        self.alive = 1
        self.click = None
        self.drag = False
        self.keystates = {}

        #self.test = vertex()
        #setup()

        self.list_of_objects = []

        self.rx = self.ry = self.rz = 0

        self.framerate = 0.010

        self.fps = 0
        self.lastmeasurepoint = time()
        self.fr = pyglet.text.Label(text='calculating fps', font_name='Verdana', font_size=8, bold=False, italic=False,
                                        color=(255, 255, 255, 255), x=10, y=8,
                                        anchor_x='left', anchor_y='baseline',
                                        multiline=False, dpi=None, batch=None, group=None)

    #def update(self, dt):
    #	self.rx += dt * 1
    #	self.ry += dt * 80
    #	self.rz += dt * 30
    #	self.rx %= 360
    #	self.ry %= 360
    #	self.rz %= 360

    def add_object(self, obj):
        self.list_of_objects.append(obj)

    def on_resize(self, width, height):
        # Override the default on_resize handler to create a 3D projection
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60., width / float(height), .1, 1000.)
        glMatrixMode(GL_MODELVIEW)
        return pyglet.event.EVENT_HANDLED

    def on_close(self):
        print('closing')
        self.alive = 0

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.click:
            self.drag = True
        self.rx += dx
        self.ry += dy
        self.rz += dy
        self.rx %= 360
        self.rz %= 360

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        glTranslatef(0, 0, scroll_y*10)

    def on_mouse_press(self, x, y, button, modifiers):
        pass #print button

    def on_mouse_release(self, x, y, button, modifiers):
        self.click = None
        self.drag = False

    def on_key_press(self, symbol, modifiers):
        if symbol == 65307:
            self.alive = 0
        elif symbol == 32:
            self.rx = self.ry = self.rz = 0
            glLoadIdentity()
            glTranslatef(0, 0, -200)
        else:
            self.keystates[symbol] = True
            #print symbol

    def on_key_release(self, symbol, modifiers):
        self.keystates[symbol] = False

    def render(self):
        #glLoadIdentity()
        #glTranslatef(0, 0, -200)
        self.clear()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #glLoadIdentity()
        #glTranslatef(0, 0, 0)
        #glRotatef(self.rz, 0, 0, 1)
        #glRotatef(self.ry, 0, 1, 0)
        #glRotatef(self.rx, 1, 0, 0)
        for obj in self.list_of_objects:
            obj.draw()


        #self.test.obj.draw(pyglet.gl.GL_LINES)
        #self.bg.blit(0,0)

    def run(self):
        while self.alive == 1:
            event = self.dispatch_events()

            if 65362 in self.keystates and self.keystates[65362] > 0:# up
                self.rx -= 0.25
                #self.rx %= 360
                glRotatef(self.rx%360, 1, 0, 0)
            if 65363 in self.keystates and self.keystates[65363] > 0:# right
                self.ry -= 0.25
                #self.ry %= 360
                glRotatef(self.ry%360, 0, 1, 0)
            if 65364 in self.keystates and self.keystates[65364] > 0:# down
                self.rx += 0.25
                #self.rx %= 360
                glRotatef(self.rx%360, 1, 0, 0)
            if 65361 in self.keystates and self.keystates[65361] > 0:# left
                self.ry += 0.25
                #self.ry %= 360
                glRotatef(self.ry%360, 0, 1, 0)
            if 122 in self.keystates and self.keystates[122] > 0: # z
                self.rz += 0.25
                glRotatef(self.rz%360, 0, 0, 1)

            self.render()
            self.flip()
            self.fps += 1
            if time() - self.lastmeasurepoint >= 1:
                self.fr.text = str(self.fps) + 'fps'
                self.fps = 0
                self.lastmeasurepoint = time()
            sleep(self.framerate)


g = gui()
g.add_object(TriMesh(vertices, normals, faces))
g.run()

'''






















'''
from version1.utilities import read_obj
import pyglet
from pyglet.gl import *
import ctypes
import numpy as np


# Zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1/ZOOM_IN_FACTOR


lightfv = ctypes.c_float * 4
rotation = 0
T = 0

vertices, faces, normals = read_obj("../obj-files/atlas.obj")
vertices = np.array(vertices)

vertices = 10.0 * (vertices - np.mean(vertices, axis = 0)) + np.array([100, 100, 3.0])


diffuse = [.8, .8, .8, 1.]
ambient = [.2, .2, .2, 1.]
specular = [0., 0., 0., 1.]
emissive = [0., 0., 0., 1.]
shininess = 0.



def gl_light(lighting):
    """Return a GLfloat with length 4, containing the 4 lighting values."""
    return (GLfloat * 4)(*(lighting))

def hextoint(i):
    if i > 255:
        i = 255
    return (1.0/255.0) * i


def setup():
    # One-time GL setup
    #glClearColor(1, 1, 1, 1)
    #glColor3f(hextoint(0), hextoint(0), hextoint(0))
    #glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    # Uncomment this line for a wireframe view
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    # Simple light setup.  On Windows GL_LIGHT0 is enabled by default,
    # but this is not the case on Linux or Mac, so remember to always
    # include it.
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)

    # Define a simple function to create ctypes arrays of floats:
    def vec(*args):
        return (GLfloat * len(args))(*args)

    glLightfv(GL_LIGHT0, GL_POSITION, vec(.5, .5, 1, 0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, vec(.5, .5, 1, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1, 1, 1, 1))
    glLightfv(GL_LIGHT1, GL_POSITION, vec(1, 0, .5, 0))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, vec(.5, .5, .5, 1))
    glLightfv(GL_LIGHT1, GL_SPECULAR, vec(1, 1, 1, 1))

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(hextoint(30), hextoint(30), hextoint(30), hextoint(255)))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 1))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 10)

    glTranslatef(0, 0, -200)



class App(pyglet.window.Window):

    def __init__(self, width, height, *args, **kwargs):
        conf = Config(sample_buffers=1,
                      samples=4,
                      depth_size=500.0,
                      double_buffer=True)
        
        super(App, self).__init__(width, height, config=conf, *args, **kwargs)

        #Initialize camera values
        self.left   = 0
        self.right  = width
        self.bottom = 0
        self.top    = height
        self.zoom_level = 1
        self.zoomed_width  = width
        self.zoomed_height = height

        setup()

    def init_gl(self, width, height):
        # Set clear color
        glClearColor(0/255, 0/255, 0/255, 0/255)

        # Set antialiasing
        glEnable( GL_LINE_SMOOTH )
        glEnable( GL_POLYGON_SMOOTH )
        glHint( GL_LINE_SMOOTH_HINT, GL_NICEST )

        # Set alpha blending
        glEnable( GL_BLEND )
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )

        # Set viewport
        glViewport( 0, 0, width, height )

    def on_resize(self, width, height):
        # Set window values
        self.width  = width
        self.height = height
        # Initialize OpenGL context
        self.init_gl(width, height)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # Move camera
        self.left   -= dx*self.zoom_level
        self.right  -= dx*self.zoom_level
        self.bottom -= dy*self.zoom_level
        self.top    -= dy*self.zoom_level

    def on_mouse_scroll(self, x, y, dx, dy):
        # Get scale factor
        f = ZOOM_IN_FACTOR if dy > 0 else ZOOM_OUT_FACTOR if dy < 0 else 1
        # If zoom_level is in the proper range
        if .2 < self.zoom_level*f < 5:

            self.zoom_level *= f

            mouse_x = x/self.width
            mouse_y = y/self.height

            mouse_x_in_world = self.left   + mouse_x*self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y*self.zoomed_height

            self.zoomed_width  *= f
            self.zoomed_height *= f

            self.left   = mouse_x_in_world - mouse_x*self.zoomed_width
            self.right  = mouse_x_in_world + (1 - mouse_x)*self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y*self.zoomed_height
            self.top    = mouse_y_in_world + (1 - mouse_y)*self.zoomed_height

    def on_draw(self):

        # Initialize Projection matrix
        #glMatrixMode( GL_PROJECTION )
        #glLoadIdentity()

        # Initialize Modelview matrix
        glMatrixMode( GL_MODELVIEW )
        glLoadIdentity()
        
        # Save the default modelview matrix
        glPushMatrix()

        # Clear window with ClearColor
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        # Draw outlines only
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Set orthographic projection matrix
        glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )

        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, gl_light(diffuse) )
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, gl_light(ambient) )
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, gl_light(specular) )
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, gl_light(emissive) )
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)


        # draw the vertebra
        glBegin(GL_TRIANGLES)
        for face in faces:
            for index in face:
                glVertex3f(vertices[index-1, 0], vertices[index-1, 1], vertices[index-1, 2])
        glEnd()


        # Remove default modelview matrix
        glPopMatrix()

    def update(self, dt):
        global vertices, T
        T += dt
        vertices += np.array([0, 0, 10*np.sin(T)]) * dt

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1/120.0)
        pyglet.app.run()


App(500, 500).run()

'''





'''
# make a new window
main_window = pyglet.window.Window(800, 400)


@main_window.event
def on_draw():
    # clear stuff    
    #glClear(GL_COLOR_BUFFER_BIT)
    
    main_window.clear()
    
    glLoadIdentity()
    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-40, 200, 100, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightfv(0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightfv(0.5, 0.5, 0.5, 1.0))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_MODELVIEW)
    
    
    #glTranslatef(0, 0.0, -0.0)
    #glRotatef(90.0, 1, 0, 0)
    glTranslated(400, 200, -0.0)
    
    glRotatef(-66.5, 0, 0, 1)
    glRotatef(rotation, 1, 0, 0)
    glRotatef(90, 0, 0, 1)
    glRotatef(0, 0, 1, 0)
    
    
     
    # draw the verticces and stuff
    glBegin(GL_TRIANGLES)
    for face in faces:
        for i in range(0,3):
            glVertex3f(vertices[face[i]-1,0], vertices[face[i]-1,1], vertices[face[i]-1,2])
    glEnd()
def update():
    global rotation
    rotation += 45*dt
    if rotation > 720:
        rotation = 0

pyglet.app.run()

'''
































