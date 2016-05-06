"""
The Utilities Module
    Jeff M. Barrett, 2016
    M.Sc. Candidate | Biomechanics
    University of Waterloo

This contains a number of functions useful for reading in data from different sources.
Right now it only reads in Wavefront object files, but there are plans to expand this
in the future to include things like c3d files, an out-of-the-box csv reader, and so on.


"""

def read_obj(filename, combine = list.append):
    """
    Purpose: Reads the vertex and face information stored in the specified
             Wavefront object file.
    Inputs: filename is the name (and directory) of the object file that one
            wishes to read from
    Outputs: returns the vertices, faces, and normals specified in the fiel
    Time: O(n) where n is the number of lines in the file
    """
    vertices = []
    faces = []
    normals = []

    f = open(filename, 'r')
    for line in f:
        if line.startswith("v"):
            data = line.split()
            if (data[0] == 'vn' and len(data) == 4):
                combine(normals, list(map(float, data[1:])))
            elif (data[0] == 'v'):
                combine(vertices, list(map(float, data[1:])))
        elif line.startswith('f'):
            data = line.split()
            triangle = []
            for face in data[1:]:
                vertexnum = face.split('/')
                triangle.append(int(vertexnum[0]))
            combine(faces, triangle)
        else:
            pass

    return vertices, faces, normals






















