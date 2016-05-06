"""
The GFX-Module
    Written by Jeff M. Barrett April 2016
    M.Sc. Candidate | Biomechanics
    University of Waterloo

This is an object-oriented module for keeping track of meshes:
    1) Triangular Meshes
    2) Line Segments

In cases where the vertices in each mesh is stored in some local coordinate system and needs to be transformed into
a global space, there are methods built-in that will deal with this.

"""
import numpy as np
import pyglet
from pyglet.gl import *
import ctypes
import version1.biomech as bm



def quatmult(r, q):
    """
    Purpose: Multiplies two quaternions (arrays with dimension 4)
    :param q1:  the first quaternion
    :param q2:  the second quaternion
    :return:
    """
    t0 = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]
    t1 = r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2]
    t2 = r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1]
    t3 = r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0]
    return np.array([t0, t1, t2, t3])


def quatinv(q):
    """
    Purpose: Returns the inverse of the provided quaternion
    :param q:
    :return:
    """
    if all(q == 0):
        print("q is not invertible!")
        return np.array([0, 0, 0, 0])
    else:
        return quatconj(q)/mag(q)


def quatconj(q):
    """
    Purpose: Computes the conjugate of the provided quaternion
    :param q:
    :return:
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def mag2(q):
    """
    Purpose: Computes the norm of the provided quaternion
    :param q:
    :return:
    """
    return sum(n * n for n in q)


def mag(q):
    """
    Returns the magnitude of the provided quaternion
    :param q:
    :return:
    """
    return np.sqrt(mag2(q))


def normalize(q):
    """
    Purpose: Normalizes the provided quaternion
    :param q:
    :param tolerance:
    :return:
    """
    return q / mag(q)




def point2arm(q, r):
    """
    The definition of a rotation about a point is (q, r) where q is the quaternion
    and r is the point. This can be re-written in axis-arm convention (I don't think
    that's the technical term, but it's one that I've made up) as:
            (q, a)
        Where q is the same quaternion and:
            a = r - q*r*conj(q)
    :param p:
    :return:
    """
    return r - quatmult(quatmult(q, r), quatconj(q))


def arm2point(q, a):
    """
    The inverse mapping of point2arm:
            a = r - q*r*conj(q)
    First we note that: conj(r) = -r, so that if we conjugate both sides we get:
            conj(a) = -r + conj(q)*r*q
    :param q:
    :param a:
    :return:
    """


def combine_rotations(rot1, rot2):
    """

    :param rot1:
    :param rot2:
    :return:
    """
    q, r = rot1
    p, s = rot2
    a = point2arm(q,r)
    b = point2arm(p,s)





class Mesh(object):

    def __init__(self, vertices = np.array([]), normals = np.array([]), faces = np.array([]), meshtype = GL_TRIANGLES):
        """

        :param vertices:            a numpy-array of vertices (Nx3)
        :param normals:             a numpy array of normals (Nx3)
        :param faces:               an array of face-connectors (Nx3)
        :return:
        """

        self.vertices = np.array(vertices)
        self.normals = np.array(normals)
        self.faces = np.array(faces)
        self.type = meshtype
        #self.colour = np.tile(np.array([0.85, 0.76, 0.51]), len(self.vertices))#0.1 * np.ones([self.vertices.size])
        self.colour = np.tile(np.array([0.1, 0.09, 0.08]), len(self.vertices))#0.1 * np.ones([self.vertices.size])
        self.set_up_list()


        # properties needed for translation and rotation using OpenGL
        self.euler_angle = np.array([0.0, 0.0, 0.0])
        self.point_of_rotation = self.centroid()
        self.translation = np.array([0.0, 0.0, 0.0])




    def set_up_list(self):
        def vec(*args):
            return (GLfloat * len(args))(*args)


        verts = self.vertices.flatten()
        vfaces = self.faces.flatten()
        vnorms = self.normals.flatten()
        vcols = self.colour.flatten()

        verts = (GLfloat * len(verts))(*verts)
        vnorms = (GLfloat * len(vnorms))(*vnorms)
        vfaces = (GLuint * len(vfaces))(*vfaces)
        vcols = (GLfloat * len(vcols))(*vcols)

        self.list = glGenLists(1)
        glNewList(self.list, GL_COMPILE)

        # set up the material properties
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, vec(0.4, 0.4, 0.4, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, vec(0.1, 0.1, 0.1, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(0.0, 0.0, 0.0, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, vec(0.2, 0.2, 0.2, 1.0))

        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glColorPointer(3, GL_FLOAT, 0, vcols)
        glVertexPointer(3, GL_FLOAT, 0, verts)
        glNormalPointer(GL_FLOAT, 0, vnorms)
        glDrawElements(GL_TRIANGLES, len(vfaces), GL_UNSIGNED_INT, vfaces)
        glPopClientAttrib()
        glEndList()


    def apply_scale(self, alpha):
        """
        Purpose: Applies the scale, alpha, to the model (while keeping the centroid constant)
        :param alpha:
        :return:
        """
        centroid = self.centroid()
        self.vertices = alpha * (self.vertices - centroid) + centroid
        self.set_up_list()

    def translate_mesh(self, dx):
        """
        Purpose: Translates the model in 3D space by amounts dx = np.array([x,y,z])
        :param x:
        :param y:
        :param z:
        :return:
        """
        self.vertices += dx
        self.point_of_rotation = self.centroid()
        self.set_up_list()


    def translate_mesh_to(self, dx):
        """
        Purpose: Moves the mesh's centroid to the point (x,y,z) then updates the list afterwards
        :param x:
        :param y:
        :param z:
        :return:
        """
        self.vertices -= self.centroid() - dx
        self.set_up_list()


    def rotate_mesh(self, phi, theta, psi):
        """
        Purpose: Rotates the mesh in accordance with the provided angles
                NOTE: This is a very inefficient way of doing this!
        :param angles:
        :return:
        """
        R = bm.angle2dcm(phi, theta, psi, "XYZ")
        self.vertices = R.dot(self.vertices)
        self.set_up_list()




    def rotate_mesh_around_point(self, phi, theta, psi, point):
        """
        Purpose: Rotates the mesh around a given point with the provided angles
                 Remark: note that the angles are in an XYZ-sequence
        :param phi:
        :param theta:
        :param psi:
        :param point:
        :return:
        """
        # how do Euler angles and rotations compose?
        self.euler_angle += np.array([phi, theta, psi])
        self.point_of_rotation = point






    def centroid(self):
        """
        Purpose: Returns the mesh's geometric centroid
        :return: Returns an nparray containing the mesh's geometric centroid
        """
        return np.mean(self.vertices, axis = 0)


    def reorient(self):
        """
        Orients the body before drawing it (optional) this also has the added bonus of reducing
        overhead
        :return:
        """
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslated(self.point_of_rotation[0], self.point_of_rotation[1], self.point_of_rotation[2])
        glRotated(self.euler_angle[0], 1.0, 0.0, 0.0)
        glRotated(self.euler_angle[1], 0.0, 1.0, 0.0)
        glRotated(self.euler_angle[2], 0.0, 0.0, 1.0)
        glTranslated(-self.point_of_rotation[0], -self.point_of_rotation[1], -self.point_of_rotation[2])
        glTranslated(self.translation[0], self.translation[1], self.translation[2])


    def orient(self):
        """
        Orients the body in 3D space (while preserving the previous orientation)
        :return:
        """
        glMatrixMode(GL_MODELVIEW)
        glTranslated(self.point_of_rotation[0], self.point_of_rotation[1], self.point_of_rotation[2])
        glRotated(self.euler_angle[0], 1.0, 0.0, 0.0)
        glRotated(self.euler_angle[1], 0.0, 1.0, 0.0)
        glRotated(self.euler_angle[2], 0.0, 0.0, 1.0)
        glTranslated(-self.point_of_rotation[0], -self.point_of_rotation[1], -self.point_of_rotation[2])
        glTranslated(self.translation[0], self.translation[1], self.translation[2])


    def draw(self):
        """
        Purpose: Draws the triangular mesh (needs a valid opengl context to work)
                 Note that this uses a ZYX-Euler Sequence
        :return:
        """
        glCallList(self.list)



























