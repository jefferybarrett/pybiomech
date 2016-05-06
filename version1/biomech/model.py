"""
model is a sub-package of python for doing biomechanical modelling. It includes:
    1) The segment class


-> Forces in this paradigm are two-ples (x, F) where x is the point-of-application and F is the force.
        --> They should also come with a reference frame... that would be important
->

"""
import version1.gfx as gfx
import version1.biomech as bm
import numpy as np










class Segment():
    """
    The Segment class is meant to represent a segment, which consists of:
    1. A mass
    2. An inertia matrix
    3. A position
    4. An orientation (stored as a matrix)
    5. Velcoities (angular and translational)

    Currently this is set-up to be more useful for inverse dynamics than foreward dynamics (for now!)
    """


    def __init__(self, name, mass, inertia, com, R, time):
        """
        Purpose: Initializes the segment with a mass, moment of inertia, centre of mass, and orientation
                 (local coordinate system)
        :param mass:            a scalar mass
        :param inertia:         a moment of inertia (3x3 matrix, in local coordinates)
        :param com:             an array of center of mass locations (Nx3)
        :param R:               an array of rotation matrices corresponding to the orientation (Nx3x3)
        :param time:            a time-vector (Nx1) of EVENLY-SPACED SAMPLES
        :return:
        """
        self.name = name
        self.m = mass       # linear mass
        self.I = inertia    # moment of inertia centered at the center of mass (simplification)
        self.com = com      # center of mass position
        self.R = R          # orientation
        self.t = time       # the time-array

        # to be computed
        self.a = []         # linear acceleration (global coordinates)
        self.v = []         # linear velocity (global coordinates)
        self.omega = []     # angular velocity (global coordinates)
        self.alpha = []     # angular accelerations

        self.parent_segment = (np.array([0,0,0]), 0 * self.com)        # the segment's parent segment(s) [I think only one would allow this to be solvable]
        self.list_of_forces = []        # list of external forces acting on the body
        self.list_of_children = []      # list of children who articulate with the body
        self.joint_reaction = []        # a list of the joint reaction forces and moments in 1:1 correspondence to those
                                        #   listed in the list_of_children

        # some to be computed now
        self.dt = np.mean(np.diff(time))    # average time-step (this does assume that all time-points are evenly spaced)
        self.sampling_rate = 1.0/self.dt    # approximate sampling rate

        self.mesh = gfx.Mesh()


    def add_mesh(self, mesh):
        """
        Adds a gfx mesh object to the model (for animation purposes)
        The gfx vertices must be in local coordinates of the defined body for this to animate correctly!
        :param mesh:
        :return:
        """
        self.mesh = mesh



    def draw_frame(self, i):
        """
        Draws the ith frame in the dataset
        :param i:
        :return:
        """
        orientation = self.R[i]
        alpha, beta, gamma = bm.dcm2angle(orientation, "XYZ")
        self.mesh.euler_angle = np.array([alpha, beta, gamma])
        self.mesh.point_of_rotation = self.parent_segment[0]
        self.mesh.orient()
        self.mesh.draw()

        for (pos, child) in self.list_of_children:
            child.draw_frame(i)




    def add_force(self, position_in_body, force_vector):
        """
        Purpose: Adds an external force the segment

        :param position_in_body:        a 3x1 vector specifying the location of the force in the body
        :param force_vector:            an Nx3x1 vector specifying the magnitude and direction of the force in the body
        :return:
        """
        self.list_of_forces.append((position_in_body, force_vector))




    def add_parent(self, position_in_child, parent):
        """

        :param position_in_parent:
        :param parent:
        :return:
        """
        self.parent_segment = (position_in_child, parent)





    def add_child(self, position_in_parent, position_in_child, child):
        """

        :param position:        the position of the linkage in the parent body
        :param child:
        :return:
        """
        self.list_of_children.append((position_in_parent, child))
        child.add_parent(position_in_child, self)



    def compute_angular(self):
        """
        Purpose: Computes the angular velocity and acceleration of the body in three-dimensional-space. This is done
                 using the formula omega x = dot(R) * transpose(R) In particular:
                        [       0,  -omega_z,  omega_y]
                        [ omega_z,         0, -omega_x]  = (dR/dt) * R^(T)
                        [-omega_y,   omega_x,        0]
        :return:
        """
        Rdot = bm.vdiff(self.R, np.mean(np.diff(self.t)))
        self.omega = np.zeros([np.size(self.t), 3])


        omega_matrix = np.einsum('nij,nkj->nik', Rdot, self.R)

        self.omega[:,0] = omega_matrix[:, 2, 1]           # omega_x
        self.omega[:,1] = omega_matrix[:, 0, 2]           # omega_y
        self.omega[:,2] = omega_matrix[:, 1, 0]           # omega_z

        self.alpha = bm.vdiff(self.omega, np.mean(np.diff(self.t)))




    def compute_linear(self):
        """
        Purpose: Computes the linear velocity and acceleration of the segment in global coordinates

        :return:
        """
        self.v = bm.vdiff(self.com, np.mean(np.diff(self.t)))
        self.a = bm.vdiff(self.v, np.mean(np.diff(self.t)))





    def resolve_forces_and_moments(self):
        """
        Purpose: Returns the net

            Straight up: this is very poorly written. This really needs to be cleaned up...

        :return:
        """
        self.compute_angular()
        self.compute_linear()
        net_force = self.m * self.a


        net_torque = np.einsum('ij,nj->ni', self.I, self.alpha) + np.cross(self.omega, np.einsum('ij,nj->ni', self.I, self.omega), axis = 1)


        for force in self.list_of_forces:
            net_force -= force[1]
            net_torque -= np.cross(force[0], force[1], axis = 1)

        self.joint_reaction = []
        for child in self.list_of_children:
            force_from_child, torque_from_child = child[1].resolve_forces_and_moments()
            self.joint_reaction.append((force_from_child, torque_from_child))
            net_force -= force_from_child
            net_torque -= (torque_from_child + np.cross(child[0], force_from_child))

        net_torque -= np.cross(self.parent_segment[0], net_force)

        return net_force, net_torque
















































