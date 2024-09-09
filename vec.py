import math
import numpy as np
from typing import Union
import xml.etree.ElementTree as ET
import ctypes
import sys
import pathlib
from svg import createCommandStack, execCommandStack
from scipy.sparse import coo_matrix
from dataclasses import dataclass
from copy import deepcopy
import re
from zipfile import ZipFile
import plistlib
import struct
import time
import os
from collections import namedtuple
import subprocess
import json
from subprocess import CalledProcessError
from matplotlib.tri import Triangulation
from matplotlib.collections import TriMesh as TM
libname = pathlib.Path().absolute() / "libs/winding.so"
winding = ctypes.CDLL(libname)
libname = pathlib.Path().absolute() / "libs/tv.so"
tv = ctypes.CDLL(libname)

MARGIN = 1

class Point():

    def __init__(self, x: float, y: float):
        self.coords = np.array([x, y], dtype=np.float64)

    @property
    def x(self):
        return self.coords[0]
    
    @x.setter
    def x(self, value):
        self.coords[0] = value

    @property
    def y(self):
        return self.coords[1]
    
    @y.setter
    def y(self, value):
        self.coords[1] = value

    def __getitem__(self, i):
        return self.coords[i]

    def __eq__(self, other):
        if isinstance(other, Point):
            return (self.coords == other.coords).all()
        elif isinstance(other, np.ndarray):
            return self.x == other[0] and self.y == other[1]
        else:
            return False

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float):
        return Point(self.x * other, self.y * other)
    
    def __rmul__(self, other: float):
        return Point(self.x * other, self.y * other)

    def __truediv__(self, other: float):
        if other == 0:
            raise ZeroDivisionError
        return Point(self.x / other, self.y / other)

    def __repr__(self):
        return "({:.2f}, {:.2f})".format(self.x, self.y)

    def dot(self, other) -> float:
        return np.dot(self.coords, other.coords)

    def norm(self) -> float:
        return np.linalg.norm(self.coords)

    def normalize(self):
        n = self.norm()
        return (self / n)
    
    def normalize_coordinates(self, bl, dx, dy, aspect, SCALE):
        # operates in-place
        self.x = ((self.x - bl.x) * (SCALE / dx)) * aspect
        self.y = ((self.y - bl.y) * (SCALE / dy))

    def crossmag(self, other, type='none') -> float:
        # magnitude of the cross product
        c = (self.x * other.y) - (self.y * other.x)
        if type == 'pos':
            return abs(c)
        elif type == 'neg':
            return -abs(c)
        else:
            return c

    def rotate(self, origin, theta: float):
        p = self - origin
        cos, sin = math.cos(theta), math.sin(theta)
        p.x, p.y = cos * p.x - sin * p.y, sin * p.x + cos * p.y
        p += origin
        return p
        
class Edge():

    def __init__(self, p1: Point, p2: Point, id=None):
        # a directed edge
        self.id = id
        self.p1 = p1
        self.p2 = p2
        self.mid = (p1 + p2) * 0.5
        self.vec = p2 - p1
        self.dipole = Point(-self.vec.y, self.vec.x) # turn left, 90 deg CCW
        self.xmax = max(p1.x, p2.x)
        self.xmin = min(p1.x, p2.x)
        self.ymax = max(p1.y, p2.y)
        self.ymin = min(p1.y, p2.y)
        self.stroke = None
        self.length = None

    def __repr__(self):
        return f"[ {self.p1}, {self.p2} ]"
    
    def calc_length(self):
        if not self.length:
            self.length = (self.p1 - self.p2).norm()
        return self.length
    
class Stroke():

    def __init__(self, edges: list[Edge], id, bb=None, start=None, end=None):
        self.edges = edges
        if bb is None:
            self.xmin = np.inf
            self.xmax = -np.inf
            self.ymin = np.inf
            self.ymax = -np.inf
            for edge in self.edges:
                self.xmin = min(self.xmin, edge.xmin)
                self.xmax = max(self.xmax, edge.xmax)
                self.ymin = min(self.ymin, edge.ymin)
                self.ymax = max(self.ymax, edge.ymax)
        else:
            self.xmin = bb[0]
            self.xmax = bb[1]
            self.ymin = bb[2]
            self.ymax = bb[3]
        for edge in self.edges:
            edge.stroke = self
        self.length = None
        self.scale = 1
        self.id = id

    def __repr__(self):
        return repr(self.edges)
            
    def calc_length(self):
        if not self.length:
            l = 0
            for edge in self.edges:
                l += edge.calc_length()
            self.length = l
        return self.length
    
    def normalize_coordinates(self, bl: Point, dx, dy, aspect, SCALE):
        self.edges[0].p1.normalize_coordinates(bl, dx, dy, aspect, SCALE)
        for e in self.edges:
            e.p2.normalize_coordinates(bl, dx, dy, aspect, SCALE)
            e.mid = (e.p1 + e.p2) * 0.5
            e.vec = e.p2 - e.p1
            e.dipole = Point(-e.vec.y, e.vec.x) # turn left
            e.xmax = max(e.p1.x, e.p2.x)
            e.xmin = min(e.p1.x, e.p2.x)
            e.ymax = max(e.p1.y, e.p2.y)
            e.ymin = min(e.p1.y, e.p2.y)
        self.xmax = ((self.xmax - bl.x) * (SCALE / dx)) * aspect
        self.ymax = ((self.ymax - bl.y) * (SCALE / dy))
        self.xmin = ((self.xmin - bl.x) * (SCALE / dx)) * aspect
        self.ymin = ((self.ymin - bl.y) * (SCALE / dy))
        self.length = 0
        for e in self.edges:
            self.length += (e.p2 - e.p1).norm()

    def get_edges(self):
        # get edge data
        # used for plotting curves
        x = np.zeros(len(self.edges) + 1)
        y = np.zeros(len(self.edges) + 1)
        x[0] = self.edges[0].p1.x
        y[0] = self.edges[0].p1.y
        for i,edge in enumerate(self.edges):
            x[i + 1] = edge.p2.x
            y[i + 1] = edge.p2.y
        return (x, y)
    
def get_barycentric_coords(tri: np.ndarray, point: Point):
    # compute barycentric coordinates
    # https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    v0 = Point(tri[1][0], tri[1][1]) - Point(tri[0][0], tri[0][1])
    v1 = Point(tri[2][0], tri[2][1]) - Point(tri[0][0], tri[0][1])
    v2 = point - Point(tri[0][0], tri[0][1])
    denom = v0.x * v1.y - v1.x * v0.y
    v = (v2.x * v1.y - v1.x * v2.y) / denom
    w = (v0.x * v2.y - v2.x * v0.y) / denom
    u = 1 - v - w

    return u, v, w
        
def in_triangle(p: Point, c1: Point, c2: Point, c3: Point) -> bool:
    return ((c2 - c1).crossmag(p - c1) > 0) and ((c3 - c2).crossmag(p - c2) > 0) and ((c1 - c3).crossmag(p - c3) > 0)

class Sketch():

    def __init__(self, filename=None, numFields=None, useTV=True, rngSeed=None, hiRes=False):
        self.path = filename
        self.strokes = []
        self.strokesMarked = []
        self.edges = []
        self.next_edge_id = 2 # start from 2 for compatibility with TRIANGLE

        self.xmin = np.inf
        self.xmax = -np.inf
        self.ymin = np.inf
        self.ymax = -np.inf

        self.trimesh = None
        self.triangulation = None
        self.eigenfunctions = None
        self.eigenvalues = None

        self.stroke_counter = 0

        self.wfields = []
        self.w_energies = []

        self.hiRes = hiRes

        if filename:
            name_segs = self.path.split('.')
            self.name = name_segs[-2].split('/')[-1].split('\_')[-1] + '.' + name_segs[-1]
            method = name_segs[-1]
            if method == 'vec' or method == 'svg':
                tree = ET.parse(filename)
            elif method == 'json':
                # Google Quick Draw
                file = open(filename)
                tree = json.load(file)
            elif method == 'note':
                zip = ZipFile(filename, 'r')
                for file in zip.filelist:
                    if 'Session.plist' in file.filename:
                        file = zip.read(file.filename)
                        break
                tree = plistlib.loads(file)
            else:
                raise ValueError("Unsupported file extension")
            self.parse_tree(tree, method)

            # self.cleanSketch() # unstable
            self.triangulate(numFields, useTV, rngSeed)
        else:
            self.name = 'DEBUG'
            print("Created empty sketch. You'll have to call triangulate() yourself!")
    
    def triangulate(self, numFields=None, useTV=True, rngSeed=None):
        """ call this to make the magic happen"""
        # margin padding

        xdelta = (self.xmax - self.xmin) * MARGIN
        ydelta = (self.ymax - self.ymin) * MARGIN

        print(f"Aspect ratio: {xdelta} {ydelta}")

        min_delta = min(xdelta, ydelta)

        self.bl = Point(self.xmin - min_delta, self.ymin - min_delta)
        self.tl = Point(self.xmin - min_delta, self.ymax + min_delta)
        self.tr = Point(self.xmax + min_delta, self.ymax + min_delta)
        self.br = Point(self.xmax + min_delta, self.ymin - min_delta)

        # slow normalization of image coordinates
        
        SCALE = 1000
        dx_true = self.br.x - self.bl.x
        dy_true = self.tr.y - self.br.y
        aspect = dx_true / dy_true
        

        for s in self.strokes:
            s.normalize_coordinates(self.bl, dx_true, dy_true, aspect, SCALE)
            s.calc_length()

        self.xmin = ((self.xmin - self.bl.x) * (SCALE / dx_true)) * aspect
        self.xmax = ((self.xmax - self.bl.x) * (SCALE / dx_true)) * aspect
        self.ymin = ((self.ymin - self.bl.y) * (SCALE / dy_true))
        self.ymax = ((self.ymax - self.bl.y) * (SCALE / dy_true))
        
        self.bl = Point(0, 0)
        self.tl = Point(0, SCALE)
        self.tr = Point(aspect * SCALE, SCALE)
        self.br = Point(aspect * SCALE, 0)
        self.center = np.array([aspect * SCALE / 2, SCALE / 2])

        # self.splitStrokes()

        # now, the edges have been finalized
        for s in self.strokes:
            self.edges += s.edges

        # I am pretty sure that they're already sorted, too
        # key_fn = (lambda edge : edge.id)
        # self.edges = sorted(s.edges, key=key_fn)

        self.length = np.sum([stroke.calc_length() for stroke in self.strokes])

        self.to_poly()

        try:
            if self.hiRes:
                subprocess.run(["./triangle/triangle", '.'.join(self.path.split('.')[:-1] + ['poly']), "-j", "-q", "-p", "-n", "-a40"], stdout = subprocess.DEVNULL, text=False, check=True)
            else:
                subprocess.run(["./triangle/triangle", '.'.join(self.path.split('.')[:-1] + ['poly']), "-j", "-q", "-p", "-n"], stdout = subprocess.DEVNULL, text=False, check=True)
        except CalledProcessError as c:
            print("TRIANGLE failed to produce a mesh for this input.")
            exit()

        self.trimesh = TriMesh(self)
        verts, tris = self.trimesh.verts, self.trimesh.tris
        self.triangulation = Triangulation(verts[:,0], verts[:,1], tris)
        # https://stackoverflow.com/questions/41596386/tripcolor-using-rgb-values-for-each-vertex
        self.mp_tri_mesh = TM(self.triangulation)
        
        t1 = time.perf_counter()

        numFields = (max(100, len(self.strokes)) if numFields is None else numFields)

        print(f"Generating {numFields} stroke configurations...")

        self.wfields = []
        self.w_energies = []
        self._orientations = []

        # configs = np.array([np.append(np.ones(len(self.strokes) - 2), [-1,-1]), np.ones(len(self.strokes))])
        # for config in configs:
        #     self.apply_stroke_scales(config)
        #     self.wfields.append(self.trimesh.calc_wn().reshape(-1,1))
        #     self.reset_stroke_scales()

        if not rngSeed is None:
            print(f"\tSetting RNG seed to {rngSeed}.")
            np.random.seed(rngSeed)
        random_scales = np.random.choice([-1,1], (numFields, len(self.strokes)))
        self._orientations = []
        for i in range(numFields):
            self.apply_stroke_scales(random_scales[i])
            self._orientations.append(random_scales[i])
            self.wfields.append(self.trimesh.calc_wn().reshape(-1, 1))
            self.reset_stroke_scales()

        t2 = time.perf_counter()
        print(f"Finished generating stroke configurations, took {t2 - t1} seconds")

        if not rngSeed is None:
            print("Resetting RNG to a random seed! This will randomize future clustering.")
            np.random.seed()

        if useTV and len(self.strokes) > 1:
            print("Sorting winding number fields by energy")
            t3 = time.perf_counter()

            energies = self.trimesh.calc_tv_c()

            order = np.argsort(np.array(energies))
            self.w_energies = np.array(energies)[order]
            self.wfields = [self.wfields[idx] for idx in order]
            self._orientations = [self._orientations[idx] for idx in order]

            print(f"\tTV ranges from {self.w_energies[0]} to {self.w_energies[-1]}\nFinished TV, took {time.perf_counter() - t3} seconds")

            wts = self.wfields

            if (not self.w_energies[0] == self.w_energies[-1]):
                ### weight each feature by
                ### (MAX_TV - this_TV) / sum
                wts = (self.w_energies[-1] - self.w_energies) / np.sum(self.w_energies)

            # apply weights
            self.wfields = [self.wfields[idx] * wts[idx] for idx in range(len(self.wfields))]

        else:
            print(f"Not using TV. {'Your sketch only has one stroke.' if len(self.strokes) == 1 else ''}")
            self.w_energies = np.ones(len(self.wfields))

        # finally, generate a "power increment"
        # based on the scale of distances in the feature space
        # take a fraction of the squared length of the feature space's bounding box diagonal 

        self.features = np.hstack(self.wfields)

        print(f"Memory: Features take up {sys.getsizeof(self.features) / (1024 ** 2)} MB. Has {len(self.trimesh.nodes)} nodes")

        bb_min = np.apply_along_axis(np.min, 0, self.features)
        bb_max = np.apply_along_axis(np.max, 0, self.features)

        self.power_inc = np.dot(bb_max - bb_min, bb_max - bb_min) / 250

        print(f"Selected {self.power_inc} as the power increment.")

        print(f"Done loading sketch with {len(self.strokes)} strokes and {len(self.edges)} edges")
            
    @property
    def numEdges(self):
        return len(self.edges)

    def find_edge(self, id: int) -> Edge:
        return self.edges[id - 2]

    def apply_stroke_scales(self, scales: np.ndarray):
        for i, stroke in enumerate(self.strokes):
            stroke.scale = scales[i]

    def reset_stroke_scales(self):
        for stroke in self.strokes:
            stroke.scale = 1

    def to_poly(self):
        # generate a .poly (and .node) from the sketch for use with Triangle
        # this generates a LOT of redundant vertices, but Triangle seems to deal OK with it
        lines_out_node = []
        lines_out_poly = []
        vertex_id = 1
        verts = []
        edges = []
        for stroke in self.strokes:
            verts.append((stroke.edges[0].p1, vertex_id))
            vertex_id += 1
            for edge in stroke.edges:
                verts.append((edge.p2, vertex_id))
                vertex_id += 1
                edges.append((verts[vertex_id - 3][1], verts[vertex_id - 2][1], edge.id))

        # put a frame around the sketch
        
        verts.append((self.bl, vertex_id)) # bottom left
        verts.append((self.tl, vertex_id + 1)) # top left
        verts.append((self.tr, vertex_id + 2)) # top right
        verts.append((self.br, vertex_id + 3)) # bottom right

        # side 0 (right)
        verts.append((self.br + (self.tr - self.br) * 0.33, vertex_id + 4))
        verts.append((self.br + (self.tr - self.br) * 0.67, vertex_id + 5))
        # side 1 (top)
        verts.append((self.tl + (self.tr - self.tl) * 0.33, vertex_id + 6))
        verts.append((self.tl + (self.tr - self.tl) * 0.67, vertex_id + 7))
        # side 2 (left)
        verts.append((self.bl + (self.tl - self.bl) * 0.33, vertex_id + 8))
        verts.append((self.bl + (self.tl - self.bl) * 0.67, vertex_id + 9))
        # side 3 (bottom)
        verts.append((self.bl + (self.br - self.bl) * 0.33, vertex_id + 10))
        verts.append((self.bl + (self.br - self.bl) * 0.67, vertex_id + 11))

        # vertex_id += 4
        vertex_id += 12

        lines_out_node.append(f"{vertex_id - 1} 2 0 1") # vertices header
        for v in verts:
            lines_out_node.append(f"{v[1]} {v[0].x} {v[0].y}")
        node_out = "\n".join(lines_out_node)

        lines_out_poly.append(f"0 2 0 1") # dummy vertices header
        # lines_out_poly.append(f"{len(edges) + 4} 1") # segments header
        lines_out_poly.append(f"{len(edges) + 12} 1") # segments header
        for i,s in enumerate(edges):
            # id should start at 2!
            lines_out_poly.append(f"{i} {s[0]} {s[1]} {s[2]}")
        # add frame boundary segments
            
        lines_out_poly.append(f"{i + 1} {vertex_id - 11} {vertex_id - 6} 1")
        lines_out_poly.append(f"{i + 2} {vertex_id - 6} {vertex_id - 5} 1")
        lines_out_poly.append(f"{i + 1} {vertex_id - 5} {vertex_id - 10} 1")
        lines_out_poly.append(f"{i + 1} {vertex_id - 10} {vertex_id - 7} 1")
        lines_out_poly.append(f"{i + 1} {vertex_id - 7} {vertex_id - 8} 1")
        lines_out_poly.append(f"{i + 1} {vertex_id - 8} {vertex_id - 9} 1")
        lines_out_poly.append(f"{i + 1} {vertex_id - 9} {vertex_id - 1} 1")
        lines_out_poly.append(f"{i + 1} {vertex_id - 1} {vertex_id - 2} 1")
        lines_out_poly.append(f"{i + 1} {vertex_id - 2} {vertex_id - 12} 1")
        lines_out_poly.append(f"{i + 1} {vertex_id - 12} {vertex_id - 4} 1")
        lines_out_poly.append(f"{i + 1} {vertex_id - 4} {vertex_id - 3} 1")
        lines_out_poly.append(f"{i + 1} {vertex_id - 3} {vertex_id - 11} 1")

        lines_out_poly.append("0") # holes header (unused)
        poly_out = "\n".join(lines_out_poly)

        f = open('.'.join(self.path.split('.')[:-1] + ['poly']), 'w')
        f.write(poly_out)
        f.close()
        f = open('.'.join(self.path.split('.')[:-1] + ['node']), 'w')
        f.write(node_out)
        f.close()

    def parse_tree(self, tree, method='vec'):
        if method == 'vec':
            root = tree.getroot()
            if root.tag != "vec":
                ValueError("Unsupported .vec file")
            for layer in root.findall("layer"):
                # examine all objects in this layer
                objects = layer.find("objects")
                for edge in objects.findall("edge"):
                    start = edge.get('startvertex')
                    if not (start is None):
                        start = int(start)
                    end = edge.get('endvertex')
                    if not (end is None):
                        end = int(end)
                    # trim ends
                    stroke = edge.get('curve')[9:-1].replace(",", " ").split(" ")
                    xy_store = []
                    pts = []

                    xmin = np.inf
                    xmax = -np.inf
                    ymin = np.inf
                    ymax = -np.inf

                    for i in range(len(stroke)):
                        if i % 3 == 0:
                            # trim these values
                            continue
                        else:
                            val = float(stroke[i])
                            if i % 3 == 1:
                                #x
                                xmin = min(xmin, val)
                                xmax = max(xmax, val)
                            else:
                                #y
                                ymin = min(ymin, val)
                                ymax = max(ymax, val)
                            xy_store.append(val)
                            if len(xy_store) == 2:
                                pts.append(Point(*xy_store))
                                xy_store = []
                    self.add_stroke(pts, (xmin, xmax, ymin, ymax))
        elif method == 'svg':
            root = tree.getroot()
            if not "svg" in root.tag:
                ValueError("Unsupported .svg file")
            for path in root.findall("ns:path", namespaces={"ns":"http://www.w3.org/2000/svg"}):
                sParams = execCommandStack(createCommandStack(path.attrib['d']))
                pts = [Point(pt[0], pt[1]) for pt in sParams[0]]
                self.add_stroke(pts)
            for path in root.findall("ns:g/ns:path", namespaces={"ns":"http://www.w3.org/2000/svg"}):
                sParams = execCommandStack(createCommandStack(path.attrib['d']))
                pts = [Point(pt[0], pt[1]) for pt in sParams[0]]
                self.add_stroke(pts)
            for path in root.findall("ns:g/ns:g/ns:path", namespaces={"ns":"http://www.w3.org/2000/svg"}):
                sParams = execCommandStack(createCommandStack(path.attrib['d']))
                pts = [Point(pt[0], pt[1]) for pt in sParams[0]]
                self.add_stroke(pts)

        elif method == 'note':
            # https://jvns.ca/blog/2018/03/31/reverse-engineering-notability-format/
            curve_data = None
            for obj in tree['$objects']:
                if isinstance(obj, dict) and 'curvespoints' in obj.keys():
                    curve_data = obj
            if curve_data is None:
                raise ValueError("Unsupported .note file")
            curve_lengths = struct.unpack(f"{len(curve_data['curvesnumpoints']) // 4}i", curve_data['curvesnumpoints'])
            points = struct.unpack(f"{len(curve_data['curvespoints']) // 4}f", curve_data['curvespoints'])
            curves = []
            counter = 0
            for l in curve_lengths:
                verts = np.array(points[counter:counter + 2 * l])
                verts = verts.reshape((verts.size // 2 , 2))
                curves.append(verts)
                counter += 2 * l
            # for curve in curves:
            #     plt.plot(curve[:,0], curve[:,1])
            # plt.show()
            for curve in curves:
                pts = []
                xmin = np.inf
                xmax = -np.inf
                ymin = np.inf
                ymax = -np.inf
                for row in curve:
                    xmin = min(xmin, row[0])
                    xmax = max(xmax, row[0])
                    ymin = min(ymin, row[1])
                    ymax = max(ymax, row[1])
                    pts.append(Point(row[0], row[1]))
                self.add_stroke(pts, (xmin, xmax, ymin, ymax))
        elif method == 'json':
            # must be from Google Quick Draw dataset
            drawing = tree['drawing']
            for stroke in drawing:
                pts = []
                xmin = np.inf
                xmax = -np.inf
                ymin = np.inf
                ymax = -np.inf
                xy = zip(stroke[0], stroke[1])
                for xy_ in xy:
                    xmin = min(xmin, xy_[0])
                    xmax = max(xmax, xy_[0])
                    ymin = min(ymin, xy_[1])
                    ymax = max(ymax, xy_[1])
                    pts.append(Point(*xy_))
                self.add_stroke(pts, (xmin, xmax, ymin, ymax))

    def add_stroke(self, pts: list[Point], bb=None):
        """
            Performs the following tasks:
            - Creates a new set of edges from the given points
            - Creates a new Stroke
            - Updates the Sketch bounding box
        """
        newEdges = []
        for i, _ in enumerate(pts[:-1]):
            if pts[i] == pts[i+1]:
                # this happens sometimes when the data is bad
                continue
            newEdges.append(Edge(pts[i], pts[i+1], self.next_edge_id))
            self.next_edge_id += 1
        newStroke = Stroke(newEdges, self.stroke_counter, bb=bb)
        self.stroke_counter += 1
        for e in newStroke.edges:
            e.stroke = newStroke
        
        self.strokes.append(newStroke)
        self.xmin = min(self.xmin, newStroke.xmin)
        self.xmax = max(self.xmax, newStroke.xmax)
        self.ymin = min(self.ymin, newStroke.ymin)
        self.ymax = max(self.ymax, newStroke.ymax)
    
    def is_boundary(self, item) -> set:
        """
        checks if an edge or point is part of the sketch boundary
        return a list of matching edges
        do not use this unless you really have to
        """
        if isinstance(item, Edge):
            matches = set()
            for s in self.strokes:
                for e in s.edges:
                    # triangulation may have split the edge.
                    # check if both points are on the edge
                    a_to_c = item.p1 - e.p1
                    edge_len = e.vec.dot(e.vec)
                    sim1 = a_to_c.dot(e.vec)
                    p1onEdge = np.isclose(e.vec.crossmag(a_to_c), 0.) and (sim1 > -1e-6) and (sim1 < edge_len + 1e-6)
                    if not p1onEdge:
                        continue
                    a_to_d = item.p2 - e.p1
                    sim2 = a_to_d.dot(e.vec)
                    p2onEdge = np.isclose(e.vec.crossmag(a_to_d), 0.) and (sim2 > -1e-6) and (sim2 < edge_len + 1e-6)
                    if not p2onEdge:
                        continue
                    matches.add(e)
            return matches
        elif isinstance(item, Point):
            matches = set()
            if not ((self.xmin <= item.x <= self.xmax) and (self.ymin <= item.y <= self.ymax)):
                return matches
            for s in self.strokes:
                for e in s.edges:
                    a_to_c = item - e.p1
                    edge_len = e.vec.dot(e.vec)
                    sim = a_to_c.dot(e.vec)
                    if (abs(e.vec.crossmag(a_to_c)) <= 1e-6) and (sim > -1e-6) and (sim < edge_len + 1e-6):
                        matches.add(e)
            return matches
        elif isinstance(item, np.ndarray):
            if item.size == 4:
                return self.is_boundary(Edge(Point(item[0], item[1]), Point(item[2], item[3])))
            elif item.size == 2:
                return self.is_boundary(Point(item[0], item[1]))
        else:
            raise ValueError(f"Input is of type {type(item)}, not np.ndarray, Edge, or Point or has wrong shape")

def bb_check(o1: Union[Edge, Stroke, Sketch, list, Point],
            o2: Union[Edge, Stroke, Sketch, list, Point]) -> bool:
    # returns True if these objects COULD intersect
    # due to a bounding-box check
    if type(o1) in [Edge, Stroke, Sketch]:
        o1_xmin = o1.xmin
        o1_ymin = o1.ymin
        o1_xmax = o1.xmax
        o1_ymax = o1.ymax
    elif isinstance(o1, list):
        # list of two Points
        o1_xmin = o1[0].x
        o1_ymin = o1[0].y
        o1_xmax = o1[1].x
        o1_ymax = o1[1].y
    elif isinstance(o1, Point):
        o1_xmin = o1.x
        o1_ymin = o1.y
        o1_xmax = o1.x
        o1_ymax = o1.y
    else:
        raise TypeError("Invalid type for o1")
    
    if type(o2) in [Edge, Stroke, Sketch]:
        o2_xmin = o2.xmin
        o2_ymin = o2.ymin
        o2_xmax = o2.xmax
        o2_ymax = o2.ymax
    elif isinstance(o2, list):
        # list of two Points
        o2_xmin = o2[0].x
        o2_ymin = o2[0].y
        o2_xmax = o2[1].x
        o2_ymax = o2[1].y
    elif isinstance(o2, Point):
        o2_xmin = o2.x
        o2_ymin = o2.y
        o2_xmax = o2.x
        o2_ymax = o2.y
    else:
        raise TypeError("Invalid type for o2")

    return not ((o1_xmin > o2_xmax) or (o1_xmax < o2_xmin) or (o1_ymin > o2_ymax) or (o1_ymax < o2_ymin))

MeshNodeFamily_Stub = namedtuple("MeshNodeFamily_Stub", "pos id on_boundary") # container for reference-free MeshNodeFamily
MeshNode_Stub = namedtuple("MeshNode_Stub", "pos global_id neighs neighs_parent parent siblings") # container for reference-free MeshNodeFamily

class MeshNodeFamily():
    # a group of vertices corresponding to the same point
    def __init__(self, pos: Point, id: int, mesh):
        self.pos = pos
        self.id = id
        self.children = []
        self.neighs = [] # the TriMesh constructor will insert these
        self.neigh_stroke_connections = []
        self.on_boundary = [] # also inserted by the TriMesh constructor
        self.on_stroke = []
        self.mesh = mesh
        self.tris = []

    def __repr__(self):
        return f"[MeshNodeFamily @ {self.pos} with {len(self.children)} children]"

    def create_stub(self):
        return MeshNodeFamily_Stub(deepcopy(self.pos), self.id, deepcopy(self.on_boundary))

    @staticmethod
    def is_between(angle, a1, a2):
        if a1 > a2:
            return not (a2 <= angle <= a1)
        else:
            return (a1 <= angle <= a2)
        
    @staticmethod
    def is_equal_angle(a1, a2):
        TOLERANCE = 1e-6
        return MeshNodeFamily.is_between(a1, (a2 - TOLERANCE) % (2 * np.pi), (a2 + TOLERANCE) % (2 * np.pi))
        
    def add(self, other, connecting_stroke):
        if (self.id == other.id):
            print(f"ERROR: tried to connect MeshNodeFamily {self.id} with itself")
            return
        if not(other in self.neighs):
            self.neighs.append(other)
            self.neigh_stroke_connections.append(connecting_stroke) 

    def split_by_neighbors(self, edgetbl):
        if len(self.on_boundary) > 0 and (not 1 in self.on_boundary):
            boundaries = []
            for neigh in self.neighs:
                if edgetbl[self.id][neigh.id] > 0:
                    boundaries.append(neigh)
            if len(boundaries) >= 2:
                # create a set of angles corresponding to each neighbor
                # create a node for each sector
                start_angles = []
                for b in boundaries:
                    edge_out = (b.pos - self.pos).normalize()
                    start_angles.append(math.acos(edge_out.x) if edge_out.y > 0 else 2 * np.pi - math.acos(edge_out.x))
                start_angles.sort()
                for i,start in enumerate(start_angles):
                    self.children.append(MeshNode(i, self, self.mesh, start, start_angles[(i + 1) % len(start_angles)]))
            else:
                # an endpoint
                self.children.append(MeshNode(0, self, self.mesh))
        else:
            self.children.append(MeshNode(0, self, self.mesh))

    def connect_subgraph(self, edgetbl):
        # call this after split_by_neighbors has been called on all the MeshNodeFamilies of the TriMesh
        for i, neigh in enumerate(self.neighs):
            edge_out = (neigh.pos - self.pos).normalize()
            angle = math.acos(edge_out.x) if edge_out.y > 0 else 2 * np.pi - math.acos(edge_out.x)
            if edgetbl[self.id][neigh.id] > 1 and (len(self.children) > 1) and (len(neigh.children) > 1):
                # boundary point to boundary point
                # two connections on either side, using sector lookup
                for child in self.children:
                    # two children will match
                    if self.is_equal_angle(child.angle_min, angle):
                        # make a connection
                        other = neigh.get_boundary_connection_from_sector(child.angle_min, 'min')
                        child.add(other, angle, self.neigh_stroke_connections[i])
                        other.add(child, (angle + np.pi) % (2 * np.pi), self.neigh_stroke_connections[i])
                    elif self.is_equal_angle(child.angle_max, angle):
                        # make a connection
                        other = neigh.get_boundary_connection_from_sector(child.angle_max, 'max')
                        child.add(other, angle, self.neigh_stroke_connections[i])
                        other.add(child, (angle + np.pi) % (2 * np.pi), self.neigh_stroke_connections[i])
            elif edgetbl[self.id][neigh.id] > 1 and (len(self.children) > 1) and (len(neigh.children) == 1):
                # boundary point to endpoint
                # connect each relevant node to this node
                for child in self.children:
                    # two children will match
                    if self.is_equal_angle(child.angle_min, angle) or self.is_equal_angle(child.angle_max, angle): 
                        # make a connection
                        other = neigh.children[0]
                        child.add(other, angle, self.neigh_stroke_connections[i])
                        other.add(child, (angle + np.pi) % (2 * np.pi), self.neigh_stroke_connections[i])
            elif edgetbl[self.id][neigh.id] > 1 and (len(self.children) == 1):
                # endpoint to boundary point
                # connect to both sectors
                # assume that someone will connect to me instead, and just continue
                continue 
            else:
                # just one connection
                me = self.get_node_from_angle(angle)
                other = neigh.get_node_from_angle((angle + np.pi) % (2 * np.pi))
                me.add(other, angle, self.neigh_stroke_connections[i])
                other.add(me, (angle + np.pi) % (2 * np.pi), self.neigh_stroke_connections[i])
                
    def get_node_from_angle(self, angle):
        # get the node that "sees" this angle
        for node in self.children:
            if self.is_between(angle, node.angle_min, node.angle_max):
                return node
        raise ValueError(f"Unable to find a node corresponding to the angle {angle}")

    def get_boundary_connection_from_sector(self, connecting_angle, type):
        # for matching a MeshNodeFamily with a neighboring MeshNodeFamily on a shared boundary
        # and finding the node that should be used to connect the two
        co_angle = (connecting_angle + np.pi) % (2 * np.pi)
        for node in self.children:
            if type == 'min':
                if self.is_equal_angle(node.angle_max, co_angle):
                    return node
            elif type == 'max':
                if self.is_equal_angle(node.angle_min, co_angle):
                    return node
        raise ValueError(f"Unable to find a boundary connection with the angle {connecting_angle}")


class MeshNode():
    # an individual vertex: may share the same location with other vertices
    def __init__(self, id: int, parent: MeshNodeFamily, mesh, angle_min = 0, angle_max = 2 * np.pi):
        self.id = id
        self.global_id = None
        self.parent = parent
        self.mesh = mesh
        self.pos = parent.pos
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.neighs = np.zeros(0, dtype=object) # need to be sorted in CCW order
        self.neigh_angles = np.zeros(0, dtype=np.float64) # angles for sorting
        self.neigh_edge_lengths = np.zeros(0, dtype=np.float64) # length of edge to each neighbor
        self.neigh_stroke_connections = np.zeros(0, dtype=np.float64) # length of edge to each neighbor
        self.tris = []

    def create_stub(self):
        return MeshNode_Stub(deepcopy(self.pos), self.global_id, [n.global_id for n in self.neighs], [n.parent.id for n in self.neighs],  self.parent.id, len(self.parent.children))

    def add(self, other, angle, stroke_connection):
        # ensures the neighbors are sorted in CCW order
        # starting at self.angle_min and ending at self.angle_max
        if (self.parent.id == other.parent.id):
            print(f"ERROR: tried to connect MeshNode {self.global_id} with {other.global_id} but both have parent {self.parent.id}")
            return
        if not self.has_neigh_table[other.global_id]:
            self.has_neigh_table[other.global_id] = 1

            stroke_connection = (-1 if stroke_connection is None else stroke_connection)

            self.neighs = np.append(self.neighs, other)

            if self.angle_min > self.angle_max:
                if MeshNodeFamily.is_equal_angle(angle, self.angle_min):
                    angle = self.angle_min
                angle -= self.angle_min
                if angle < 0:
                    angle += 2 * np.pi

            self.neigh_angles = np.append(self.neigh_angles, angle)
            self.neigh_stroke_connections = np.append(self.neigh_stroke_connections, stroke_connection)
            e_len = (other.parent.pos - self.parent.pos).norm()
            self.neigh_edge_lengths = np.append(self.neigh_edge_lengths, e_len)

            order = np.argsort(self.neigh_angles)
            self.neigh_angles = self.neigh_angles[order]
            self.neighs = self.neighs[order]
            self.neigh_stroke_connections = self.neigh_stroke_connections[order]
            self.neigh_edge_lengths = self.neigh_edge_lengths[order]

@dataclass
class DualNode:
    id: int
    neighs: np.ndarray
    weights: np.ndarray

class TriMesh():
    """ 
    a fascimile of a proper triangle mesh 
    read from a set of .ele, .node, and .poly files, output by Triangle
    """

    def _load_verts(self, string: str):
        # 
        # Load (new!) vertices 
        #

        params = re.split(' +', string[0])
        if params[0] == '':
                params = params[1:]
        self.node_families = np.zeros(int(params[0]), dtype=object)

        for i,line in enumerate(string[1:]):
            if i >= self.node_families.shape[0]:
                break
            params = re.split(' +', line)
            if params[0] == '':
                params = params[1:]
            pos = Point(float(params[1]), float(params[2]))

            self.node_families[i] = MeshNodeFamily(pos, i, self)
    
    def _load_verts_og(self, string: str):
        #
        # Load original vertices
        #
        
        params = re.split(' +', string[0])
        if params[0] == '':
                params = params[1:]
        self.verts_og = np.zeros(int(params[0]), dtype=Point)

        for i,line in enumerate(string[1:]):
            if i >= self.node_families.shape[0]:
                break
            params = re.split(' +', line)
            if params[0] == '':
                params = params[1:]
            pos = Point(float(params[1]), float(params[2]))
            self.verts_og[i] = pos

    def _load_edges_og(self, string: str):
        #
        # Load original set of un-split edges
        #

        # identify nodes belonging to each edge of the original PSLG

        params = re.split(' +', string[1])
        if params[0] == '':
                params = params[1:]
        end = int(params[0])

        self.directed_edges_og = np.zeros((end, 2), dtype=np.int32)

        for i,line in enumerate(string[2:]):
            if i >= end:
                break
            params = re.split(' +', line)
            if params[0] == '':
                params = params[1:]
            # Record original orientation of un-split edges
            if int(params[3]) <= 1:
                continue
            self.directed_edges_og[int(params[3])] = [int(params[1]) - 1, int(params[2]) - 1]

    def _load_edges(self, string: str):
        #
        # Load new set of edges
        #

        id_to_handle = lambda x: self.node_families[int(x) - 1] if int(x) >= 1 else None

        params = re.split(' +', string[1])
        if params[0] == '':
                params = params[1:]
        end = int(params[0])

        self.directed_edges = []
        # do these two NodeFamilies share an edge that is on a stroke???? find out here
        self.connecting_stroke_table = np.ones((len(self.node_families), len(self.node_families)), dtype=np.int32) * -1

        for i,line in enumerate(string[2:]):
            if i >= end:
                break
            params = re.split(' +', line)
            if params[0] == '':
                params = params[1:]
            self.edge_boundary_table[int(params[1]) - 1][int(params[2]) - 1] = int(params[3])
            self.edge_boundary_table[int(params[2]) - 1][int(params[1]) - 1] = int(params[3])

            # tell each MeshNodeFamily the following:
            #   - what strokes are used to connect me to my neighbors, if any?
            #   - what strokes am I on, myself?

            if int(params[3]) > 1:
                corresponding_edge = self.sketch.find_edge(int(params[3]))
            else:
                corresponding_edge = None
                # continue

            connecting_stroke = (None if corresponding_edge is None else corresponding_edge.stroke.id)

            (id_to_handle(params[1])).on_boundary.append(int(params[3]))
            (id_to_handle(params[2])).on_boundary.append(int(params[3]))

            (id_to_handle(params[1])).on_stroke.append(connecting_stroke)
            (id_to_handle(params[2])).on_stroke.append(connecting_stroke)

            if not connecting_stroke is None:
                self.connecting_stroke_table[min(int(params[1]) - 1, int(params[2]) - 1)][max(int(params[1]) - 1, int(params[2]) - 1)] = connecting_stroke

            if int(params[3]) > 1:
                self.directed_edges.append(self.orient_edge([int(params[1]) - 1, int(params[2]) - 1, int(params[3])])) # fix the edge orientation

        self.directed_edges = np.array(self.directed_edges)


    def _load_triangles(self, string: str):
        # 
        # Load triangles
        #

        id_to_handle = lambda x: self.node_families[int(x) - 1] if int(x) >= 1 else None

        params = re.split(' +', string[0])
        if params[0] == '':
            params = params[1:]
        self.tris = np.zeros((int(params[0]), 3), dtype=np.int64)
        end = int(params[0])

        for i,line in enumerate(string[1:]):
            if i >= end:
                break
            params = re.split(' +', line)
            if params[0] == '':
                params = params[1:]
            
            # in_directions: 0 for outward, 1 for inward
            (id_to_handle(params[1])).add(id_to_handle(params[2]), 0)
            (id_to_handle(params[1])).add(id_to_handle(params[3]), 1)
            (id_to_handle(params[2])).add(id_to_handle(params[1]), 1)
            (id_to_handle(params[2])).add(id_to_handle(params[3]), 0)
            (id_to_handle(params[3])).add(id_to_handle(params[1]), 0)
            (id_to_handle(params[3])).add(id_to_handle(params[2]), 1)

            self.tris[i] = np.array([int(params[1]) - 1, int(params[2]) - 1, int(params[3]) - 1])

    def _split_all_nfs(self):
        for nodeFamily in self.node_families:
            if nodeFamily: 
                nodeFamily.on_boundary = list(set(nodeFamily.on_boundary))
                nodeFamily.on_stroke = list(set(nodeFamily.on_stroke))
                nodeFamily.split_by_neighbors(self.edge_boundary_table)

    def _create_flat_node_IDs(self):
        # create a flat list of nodes with unique IDs
        # count
        countNodes = 0
        for nodeFamily in self.node_families:
            if nodeFamily: 
                if (1 in nodeFamily.on_boundary):
                    self.nfs_on_frame.append(nodeFamily)
                countNodes += len(nodeFamily.children)

        self.nodes = np.zeros(countNodes, dtype=object)

        i = 0
        for nodeFamily in self.node_families:
            if not nodeFamily: continue
            for node in nodeFamily.children:
                node.has_neigh_table = np.zeros(countNodes, dtype=np.int32)
                node.global_id = i
                self.nodes[i] = node
                i += 1

    def _connect_subgraph(self):
        for nodeFamily in self.node_families:
            if nodeFamily: nodeFamily.connect_subgraph(self.edge_boundary_table)


    def _build_verts_array(self):
        """
        
            Also identifies the "corner" vertices

        """
        self.verts = np.zeros((len(self.nodes), 2), dtype=np.float64) # some will be duplicated; this is necessary
        self.tr = None
        for node in self.nodes:
            if node.pos == self.sketch.tr:
                self.tr = node
            elif node.pos == self.sketch.tl:
                self.tl = node
            elif node.pos == self.sketch.bl:
                self.bl = node
            elif node.pos == self.sketch.br:
                self.br = node
            self.verts[node.global_id] = node.pos.coords

    def _reindex_triangles(self):
        tris_to_delete = []
        self.tris_og = np.copy(self.tris)
        for i,tri in enumerate(self.tris):
            # each triangle contains references to NodeFamilies
            # update the entry with references to actual nodes instead
            # because you do the reindexing here, the vertex weights should be correct
            family_verts = tri.copy()
            for j,fv in enumerate(family_verts):
                others = np.delete(family_verts, j)
                candidates = []
                # find the node within this family
                # that can "see" all the other families
                found = False

                if len(self.node_families[fv].children) == 1:
                    node = self.node_families[fv].children[0]
                    node.tris.append(i)
                    node.parent.tris.append(i)
                    self.tris[i][j] = node.global_id
                    found = True
                    continue

                for node in self.node_families[fv].children:
                    neighs = [neigh.parent.id for neigh in node.neighs]
                    p1 = None
                    p2 = None
                    for neigh in node.neighs:
                        if neigh.parent.id == others[0]:
                            p1 = neigh
                        elif neigh.parent.id == others[1]:
                            p2 = neigh
                    if not (p1 is None or p2 is None):
                        # this MAY be the right node....
                        # the angle of the bisector needs to be in the right range
                        bisector = (((p1.parent.pos + p2.parent.pos) / 2) - node.parent.pos).normalize()
                        angle = math.acos(bisector.x) if bisector.y > 0 else 2 * np.pi - math.acos(bisector.x)
                        if MeshNodeFamily.is_between(angle, node.angle_min, node.angle_max):
                            # found the right node
                            node.tris.append(i)
                            node.parent.tris.append(i)
                            self.tris[i][j] = node.global_id
                            found = True
                            break
                        else:
                            candidates.append(node)
                if not found:
                    # if we're here, we didn't find an appropriate node...
                    if not candidates:
                        raise ValueError(f"Couldn't find node in family {fv} for triangle {tri}")
                        print(f"WARNING: Couldn't find node in family {fv} for triangle {tri}, and no candidates exist! We will try to continue by deleting this triangle...")
                        tris_to_delete.append(i)
                    elif len(candidates) == 1:
                        print(f"WARNING: Couldn't find node in family {fv} for triangle {tri}, but only one candidate exists: falling back")
                        candidates[0].tris.append(i)
                        candidates[0].parent.tris.append(i)
                        self.tris[i][j] = candidates[0].global_id
                    else:
                        print(f"WARNING: Couldn't find node in family {fv} for triangle {tri}, and multiple candidates exist: falling back on first one encountered")
                        candidates[0].tris.append(i)
                        candidates[0].parent.tris.append(i)
                        self.tris[i][j] = candidates[0].global_id
        
        if tris_to_delete:
            print("WARNING: Deleting triangles! This might be bad!")
            self.tris = np.delete(self.tris, tris_to_delete, axis=0)

    def _build_edge_adjacency(self):
        # enumerate all the re-indexed edges
        self.my_adjacency = np.zeros((self.nodes.shape[0], self.nodes.shape[0]))
        for i, tri in enumerate(self.tris):
            tri = np.sort(tri)
            p0 = tri[0]
            p1 = tri[1]
            p2 = tri[2]
            self.my_adjacency[p0][p1] = 1
            self.my_adjacency[p1][p2] = 1
            self.my_adjacency[p0][p2] = 1

        self.my_edges = np.argwhere(self.my_adjacency > 0)

    def _calc_graph_quantites(self):
        self.edge_weights = np.zeros((len(self.verts), len(self.verts)), dtype=np.float64)
        self.vertex_weights = np.zeros(len(self.verts), dtype=np.float64)
        self.edge_lengths = np.zeros((len(self.verts), len(self.verts)), dtype=np.float64)
        self.tri_areas = np.zeros(len(self.tris), dtype=np.float64)
        self.tri_centers = np.zeros((len(self.tris), 2), dtype=np.float64)
        for i, tri in enumerate(self.tris):
            p0 = self.nodes[tri[0]].pos
            p1 = self.nodes[tri[1]].pos
            p2 = self.nodes[tri[2]].pos
            self.tri_centers[i] = ((p0.coords + p1.coords + p2.coords) / 3)
            # calc cotan contributions and add to each edge (vertex pair)
            # also calc area contribution and add to each vertex
            d0 = (p0 - p2)
            d1 = (p1 - p2)

            self.edge_lengths[tri[0]][tri[1]] = np.linalg.norm(p0.coords - p1.coords)
            self.edge_lengths[tri[1]][tri[2]] = np.linalg.norm(d1.coords)
            self.edge_lengths[tri[2]][tri[0]] = np.linalg.norm(d0.coords)

            if np.isclose(d0.crossmag(d1, 'pos'), 0):
                # degenerate triangle
                print("WARNING: Degenerate triangle: ", p0, p1, p2)
                raise ValueError("Your input created a degenerate triangle.")
                # continue
            area = d0.crossmag(d1, 'pos') / 2
            self.tri_areas[i] = area
            area_third = area / 3
            self.vertex_weights[tri[0]] += area_third
            self.vertex_weights[tri[1]] += area_third
            self.vertex_weights[tri[2]] += area_third
            d0 = d0.normalize()
            d1 = d1.normalize()
            self.edge_weights[tri[0]][tri[1]] += d0.dot(d1) / d0.crossmag(d1, 'pos')
            d0 = (p1 - p0).normalize()
            d1 = (p2 - p0).normalize()
            self.edge_weights[tri[1]][tri[2]] += d0.dot(d1) / d0.crossmag(d1, 'pos')
            d0 = (p0 - p1).normalize()
            d1 = (p2 - p1).normalize()
            self.edge_weights[tri[2]][tri[0]] += d0.dot(d1) / d0.crossmag(d1, 'pos')

    def _create_dual_graph(self, string):
        # finally, create the dual graph
            
        params = re.split(' +', string[0])
        if params[0] == '':
            params = params[1:]
        self.tri_graph = np.zeros(int(params[0]), dtype=object)
        end = int(params[0])
            
        for i,line in enumerate(string[1:]):
            if i >= end:
                break
            params = re.split(' +', line)
            if params[0] == '':
                params = params[1:]

            neighs = []
            els = []

            for j, tri_neigh in enumerate(params[1:]):
                tri_neigh = int(tri_neigh) - 1

                if tri_neigh < 0:
                    continue

                # this is a quick fix for connectivity, but it is probably slow
                if np.unique((self.tris[i], self.tris[tri_neigh])).size != 4:
                    continue

                neighs.append(tri_neigh)
                
                if j == 0:
                    els.append(self.edge_lengths[self.tris[i][1]][self.tris[i][2]]) # across from 0
                elif j == 1:
                    els.append(self.edge_lengths[self.tris[i][0]][self.tris[i][2]])
                else:
                    els.append(self.edge_lengths[self.tris[i][0]][self.tris[i][1]])
            
            self.tri_graph[i] = DualNode(i, np.array(neighs), np.array(els))

    def __init__(self, sketch: Sketch):
        
        t_start = time.perf_counter()

        self.sketch = sketch
        self.vertex_weights = None
        self.edge_weights = None
        verts_string = self.open_triangle_data(sketch.path, '1.node')
        tris_string = self.open_triangle_data(sketch.path, '1.ele')
        poly_string = self.open_triangle_data(sketch.path, '1.poly')
        trineighs_string = self.open_triangle_data(sketch.path, '1.neigh')
        # Also read the original .poly/.node files that were used to triangulate the sketch
        poly_string_og = self.open_triangle_data(sketch.path, 'poly')
        verts_string_og = self.open_triangle_data(sketch.path, 'node')

        self.boundary_vertex_ids = []
        self.nfs_on_frame = []

        t_done_txt_load = time.perf_counter()

        print(f"\tLoad text files: {t_done_txt_load - t_start} s")

        self._load_verts(verts_string)
        self._load_verts_og(verts_string_og)
        self.edge_boundary_table = np.zeros((len(self.node_families), len(self.node_families)), dtype=np.int64)
        self._load_edges_og(poly_string_og)
        self._load_edges(poly_string)
        self._load_triangles(tris_string)

        t_done_obj_creation = time.perf_counter()

        print(f"\tCreate objects: {t_done_obj_creation - t_done_txt_load} s")

        self._split_all_nfs()

        t_done_split_nfs = time.perf_counter()

        print(f"\tSplit node families: {t_done_split_nfs - t_done_obj_creation} s")

        self._create_flat_node_IDs()

        t_done_node_reindex = time.perf_counter()

        print(f"\tReindex nodes: {t_done_node_reindex - t_done_split_nfs} s")

        self._connect_subgraph()

        t_done_connect_subgraph = time.perf_counter()

        print(f"\tConnect subgraph: {t_done_connect_subgraph - t_done_node_reindex} s")

        self._build_verts_array()
        self._reindex_triangles()

        t_done_reindex_tris = time.perf_counter()

        print(f"\tReindex triangles: {t_done_reindex_tris - t_done_connect_subgraph} s")

        self._build_edge_adjacency()

        t_done_adjacency = time.perf_counter()

        print(f"\tBuild adjacency matrix: {t_done_adjacency - t_done_reindex_tris} s")

        self._calc_graph_quantites()

        t_done_quantities = time.perf_counter()

        print(f"\tCompute graph quantities: {t_done_quantities - t_done_adjacency} s")

        self._create_dual_graph(trineighs_string)

        t_done_dual = time.perf_counter()

        print(f"\tCreate dual graph: {t_done_dual - t_done_quantities} s")

        print(f"Done triangulating sketch, took {t_done_dual - t_start} s")

        t2 = time.perf_counter()

        print(f"Calculating winding number interactions...")

        self.winding = self.calc_wn_interactions_C()

        self.recalc_wn_at_endpoints()

        print(f"Done calculating winding number interactions, took {time.perf_counter() - t2} s")


    def recalc_wn_at_endpoints(self):
        """
            Fix the per-stroke winding numbers with the following strategies:

            For node in nodes:

                If node is on ANY stroke:
                    For stroke in strokes:

                        If node belongs to the current stroke under consideration:
                            If node has valence > 2:
                                Replace with average
                            Elif node is just on this stroke, and no others:
                                If node is endpoint:
                                    Replace with average
                                Else:
                                    Do nothing, it is correct
                            Else, we must be a vertex that connects two different strokes end-to-end:
                                Replace with average
                        Else:
                            Do nothing, it is correct

        """

        self.nodes_per_stroke_changed = [] # for debugging
        for node_idx, node in enumerate(self.nodes):
            if len(node.parent.on_boundary) > 0:
                for s_idx, s_field in enumerate(self.winding):
                    if s_idx in node.parent.on_stroke:
                        if len(node.parent.children) > 2:
                            self.winding[s_idx][node_idx] = self.get_avg_wn_from_neighs(node, s_field)
                            self.nodes_per_stroke_changed.append((s_idx, node_idx))
                        elif len(node.parent.on_stroke) == 1:
                            if len(node.parent.children) == 1:
                                self.winding[s_idx][node_idx] = self.get_avg_wn_from_neighs(node, s_field)
                                self.nodes_per_stroke_changed.append((s_idx, node_idx))
                        else:
                            self.winding[s_idx][node_idx] = self.get_avg_wn_from_neighs(node, s_field)
                            self.nodes_per_stroke_changed.append((s_idx, node_idx))

    def in_triangle(self, p: Union[np.ndarray, Point], tri):

        if isinstance(p, np.ndarray):
            p = Point(p[0], p[1])
        
        c1 = self.nodes[self.tris[tri][0]].pos
        c2 = self.nodes[self.tris[tri][1]].pos
        c3 = self.nodes[self.tris[tri][2]].pos

        return ((c2 - c1).crossmag(p - c1) > 0) and ((c3 - c2).crossmag(p - c2) > 0) and ((c1 - c3).crossmag(p - c3) > 0)
    
    def get_barycentric_coords(self, pt: Union[np.ndarray, Point], tri):
        # returns barycentric coordinates if pt is in tri
        # or None

        if isinstance(pt, Point):
            pt = pt.coords

        # compute barycentric coordinates
        # https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
        p0 = self.verts[self.tris[tri][0]]
        p1 = self.verts[self.tris[tri][1]]
        p2 = self.verts[self.tris[tri][2]]
        v0 = p1 - p0
        v1 = p2 - p0
        v2 = pt - p0
        denom = v0[0] * v1[1] - v1[0] * v0[1]
        v = (v2[0] * v1[1] - v1[0] * v2[1]) / denom
        w = (v0[0] * v2[1] - v2[0] * v0[1]) / denom
        u = 1 - v - w

        return [u, v, w]

    def get_triangle(self, pt: Union[np.ndarray, Point]):
        # returns the triangle in which a point is located,
        # AND its barycentric coordinates

        if isinstance(pt, Point):
            pt = pt.coords

        # find closest point
        dists = np.multiply(self.verts - pt, self.verts - pt).sum(1)
        order = np.argsort(dists)

        for next in order:
            closest_pt = self.nodes[next]

            # iterate over tris
            for tri in closest_pt.parent.tris:
                if self.in_triangle(pt, tri):
                    return tri, self.get_barycentric_coords(pt, tri)
        
        raise ValueError("Couldn't match point to triangle")

    def get_avg_wn_from_neighs(self, node, s_field):
        denom = 0
        approx = 0
        for neigh_idx, neigh in enumerate(node.neighs):
            recip = (1 / node.neigh_edge_lengths[neigh_idx])
            denom += recip
            approx += s_field[neigh.global_id] * recip
        return approx / denom

    def open_triangle_data(self, path, extension):
        path_base = '.'.join(path.split('.')[:-1])
        f = open(path_base + '.' + extension)
        f_lines = f.read().split('\n')
        f.close()

        os.remove(path_base + '.' + extension)
        return f_lines

    def dump_graph_data(self):
        # dumps self.nodes and self.node_families in a format
        # that is hopefully reference-free enough to be serializable
        nodes = np.zeros(self.nodes.shape, dtype=object)
        for i, node in enumerate(self.nodes):
            nodes[i] = node.create_stub()

        node_families = np.zeros(self.node_families.shape, dtype=object)
        for i, node_family in enumerate(self.node_families):
            node_families[i] = node_family.create_stub()

        return (nodes, node_families)
        
    def orient_edge(self, e_data):
        # orients a sub-edge
        # with respect to the original orientation of the reference edge
        p1 = self.node_families[e_data[0]].pos
        p2 = self.node_families[e_data[1]].pos
        ref_edge = e_data[2]
        p0 = self.verts_og[self.directed_edges_og[ref_edge][0]]
        
        t1 = (p1 - p0).norm()
        t2 = (p2 - p0).norm()
        if t1 > t2:
            return (e_data[1], e_data[0], ref_edge)
        else:
            return (e_data[0], e_data[1], ref_edge)

    def calc_wn_interactions_C(self):
        # wrapper for C implementation
        
        num_strokes = len(self.sketch.strokes)
        stroke_bbs = np.zeros(num_strokes * 4, dtype=np.float64)
        stroke_approx = np.zeros(num_strokes * 4, dtype=np.float64)
        for i, stroke in enumerate(self.sketch.strokes):
            stroke_bbs[4 * i] = stroke.xmin
            stroke_bbs[4 * i + 1] = stroke.ymin
            stroke_bbs[4 * i + 2] = stroke.xmax
            stroke_bbs[4 * i + 3] = stroke.ymax
            stroke_approx[4 * i] = stroke.edges[0].p1.x
            stroke_approx[4 * i + 1] = stroke.edges[0].p1.y
            stroke_approx[4 * i + 2] = stroke.edges[-1].p2.x
            stroke_approx[4 * i + 3] = stroke.edges[-1].p2.y
            
        num_edges = len(self.directed_edges)
        stroke_ids = np.zeros(num_edges, dtype=np.int32)
        all_edges = np.zeros(num_edges * 2, dtype=np.int32)
        for i, edge in enumerate(self.directed_edges):
            stroke_id = self.sketch.find_edge(edge[2]).stroke.id
            stroke_ids[i] = stroke_id
            all_edges[2 * i] = edge[0]
            all_edges[2 * i + 1] = edge[1]

        num_vertices = len(self.nodes)
        parents = np.zeros(num_vertices, dtype=np.int32)
        siblings_count = np.zeros(num_vertices, dtype=np.int32)
        neighs_parent = np.zeros(num_vertices * 2, dtype=np.int32)
        vertices = np.zeros(num_vertices * 2, dtype=np.float64)
        angle_min = np.zeros(num_vertices, dtype=np.float64)
        angle_max = np.zeros(num_vertices, dtype=np.float64)
        for i, node in enumerate(self.nodes):
            parents[i] = node.parent.id
            siblings_count[i] = len(node.parent.children)
            neighs_parent[2 * i] = node.neighs[0].parent.id
            neighs_parent[2 * i + 1] = node.neighs[-1].parent.id
            vertices[2 * i] = node.pos.x
            vertices[2 * i + 1] = node.pos.y
            angle_min[i] = node.angle_min
            angle_max[i] = node.angle_max

        num_vertices_nf = len(self.node_families)
        vertices_nf = np.zeros(num_vertices_nf * 2, dtype=np.float64)
        for i, nf in enumerate(self.node_families):
            if not nf: continue
            vertices_nf[2 * i] = nf.pos.x
            vertices_nf[2 * i + 1] = nf.pos.y

        winding.winding.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(num_strokes * num_vertices,))
        results = winding.winding(vertices.ctypes.data_as(ctypes.c_void_p), num_vertices,
                        all_edges.ctypes.data_as(ctypes.c_void_p), num_edges, 
                        stroke_ids.ctypes.data_as(ctypes.c_void_p), stroke_bbs.ctypes.data_as(ctypes.c_void_p), 
                        stroke_approx.ctypes.data_as(ctypes.c_void_p), num_strokes,
                        vertices_nf.ctypes.data_as(ctypes.c_void_p), num_vertices_nf,
                        parents.ctypes.data_as(ctypes.c_void_p),
                        siblings_count.ctypes.data_as(ctypes.c_void_p),
                        neighs_parent.ctypes.data_as(ctypes.c_void_p),
                        angle_min.ctypes.data_as(ctypes.c_void_p),
                        angle_max.ctypes.data_as(ctypes.c_void_p))
        
        results = results.reshape(num_strokes, num_vertices)
        return results
    
    def calc_face_gradients(self, scalar_map):
        """
            Calculate gradients over the faces of the triangulation

            scalar_map: flat array of (# nodes) values
        """
        grads = np.zeros((self.tris.shape[0], 2), dtype=np.float64)
        for i, tri in enumerate(self.tris):

            idx_i, idx_j, idx_k = tri[0], tri[1], tri[2]

            p_i = Point(self.verts[idx_i][0], self.verts[idx_i][1])
            p_j = Point(self.verts[idx_j][0], self.verts[idx_j][1])
            p_k = Point(self.verts[idx_k][0], self.verts[idx_k][1])

            e_ki = Edge(p_k, p_i)
            e_ij = Edge(p_i, p_j)

            tri_area = self.tri_areas[i]
            grad = (scalar_map[idx_j] - scalar_map[idx_i]) * (e_ki.dipole.coords / (2 * tri_area)) + \
                        (scalar_map[idx_k] - scalar_map[idx_i]) * (e_ij.dipole.coords / (2 * tri_area))
            grads[i] = grad
        return grads
    
    def calc_tv_c(self):
        # wrapper for C implementation

        num_tris = self.tris.shape[0]
        tris = np.zeros(num_tris * 3, dtype=np.int32)
        tri_areas = np.zeros(num_tris, dtype=np.float64)
        for i, tri in enumerate(self.tris):
            tris[3 * i] = tri[0]
            tris[3 * i + 1] = tri[1]
            tris[3 * i + 2] = tri[2]
            tri_areas[i] = self.tri_areas[i]

        num_vertices = len(self.nodes)
        vertices = np.zeros(num_vertices * 2, dtype=np.float64)
        for i, node in enumerate(self.nodes):
            vertices[2 * i] = node.pos.x
            vertices[2 * i + 1] = node.pos.y

        num_funs = len(self.sketch.wfields)
        v_funs_d = np.zeros(num_funs * num_vertices, dtype=np.float64)
        for j, fun in enumerate(self.sketch.wfields):
            for i in range(num_vertices):
                v_funs_d[j * num_vertices + i] = fun[i]


        tv.tv.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(num_funs,))
        energies = tv.tv(   tris.ctypes.data_as(ctypes.c_void_p), tri_areas.ctypes.data_as(ctypes.c_void_p), num_tris,
                            vertices.ctypes.data_as(ctypes.c_void_p), num_vertices,
                            v_funs_d.ctypes.data_as(ctypes.c_void_p), num_funs
                         )
                
        return energies
    
    def calc_tv(self, scalar_map):
        """
            sum_{tri in tris} L1(|grad|) * area

            scalar_map: flat array of (# nodes) values
        """
        sum = 0.
        grads = self.calc_face_gradients(scalar_map)
        contribs = []
        for i, tri in enumerate(self.tris):
            tri_area = self.tri_areas[i]
            contrib = np.linalg.norm(grads[i]) * tri_area
            contribs.append(contrib)
            sum += contrib
        return sum, contribs


    def calc_wn(self, stroke_idx=None):
        
        res = np.zeros(len(self.nodes), dtype=np.float64)

        if (not stroke_idx is None):
            return self.winding[stroke_idx] * (-1 if self.sketch.strokes[stroke_idx].scale < 0 else 1)
        else:
            for i, stroke in enumerate(self.sketch.strokes):
                res += self.winding[i] * (-1 if stroke.scale < 0 else 1)
            return res 

    def get_sparse_connectivity_matrix(self):
        row = []
        col = []
        data = []
        for i,node in enumerate(self.nodes):
            neighs = len(node.neighs)
            row.append(i)
            col.append(i)
            data.append(neighs)
            for neigh in node.neighs:
                row.append(i)
                col.append(neigh.global_id)
                data.append(-1)
        mat = coo_matrix((data, (row, col)), shape=(len(self.verts), len(self.verts))).tocsr().asfptype()
        return mat

    def get_sparse_cotan_laplacian(self, mode='eigen', constrained=[]):
        # if type='eigen', returns M and D^{-1}
        # else, returns L = DM
        if mode == 'eigen':
            D_inv_row = []
            D_inv_col = []
            D_inv_data = []
        else: 
            D_row = []
            D_col = []
            D_data = []
        M_row = []
        M_col = []
        M_data = []
        for i,node in enumerate(self.nodes):
            M_row.append(i)
            M_col.append(i)
            M_data.append((np.sum(self.edge_weights[i]) + np.sum(self.edge_weights[:,i])))
            if mode == 'eigen':
                D_inv_row.append(i)
                D_inv_col.append(i)
                D_inv_data.append(2 * self.vertex_weights[i])
            else:
                D_row.append(i)
                D_col.append(i)
                D_data.append(1 / (2 * self.vertex_weights[i]))
            for neigh in node.neighs:
                M_row.append(i)
                M_col.append(neigh.global_id)
                M_data.append(-(self.edge_weights[i][neigh.global_id] + self.edge_weights[neigh.global_id][i]))
        M = coo_matrix((M_data, (M_row, M_col)), shape=(len(self.verts), len(self.verts))).tocsc().asfptype()
        if mode == 'eigen':
            D_inv = coo_matrix((D_inv_data, (D_inv_row, D_inv_col)), shape=(len(self.verts), len(self.verts))).tocsc().asfptype()
            return (M, D_inv)
        else:
            D = coo_matrix((D_data, (D_row, D_col)), shape=(len(self.verts), len(self.verts))).tocsc().asfptype()
            L = D.dot(M)
            if constrained:
                lil = L.tolil()
                for c in constrained:
                    for j in range(len(self.verts)):
                        lil[c,j] = (1. if c==j else 0.)
                L = lil.tocsc()
            return L
    
    def get_single_connected_from_mask(self, fill: np.ndarray, n: float, seed: np.ndarray):
        # using a set of "filled" vertices
        # get only the vertices that are directly connected to the seed point
        new = np.zeros(fill.shape, dtype=np.float64)
        indices = np.flatnonzero(fill > n)
        discovered = []
        component = []
        queue = [seed[0]]
        while len(queue) > 0:
            next = queue.pop()
            for neigh in self.nodes[next].neighs:
                if neigh.global_id in discovered:
                    continue
                if neigh.global_id in indices:
                    queue.append(neigh.global_id)
                discovered.append(neigh.global_id)
                component.append(neigh.global_id)

        for idx in component:
            new[idx] = fill[idx]
        
        return new