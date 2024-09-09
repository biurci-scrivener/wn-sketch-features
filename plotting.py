from copy import deepcopy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.collections import TriMesh as TM, PolyCollection
import matplotlib.pyplot as plt
import numpy as np
from vec import Sketch, TriMesh
from kmeans import KMeans
import time
from PyQt5.QtWidgets import QColorDialog

KELLY_COLORS = np.array([   [0.65098039, 0.74117647, 0.84313725],
                            [0.        , 0.49019608, 0.20392157],
                            [0.75686275, 0.        , 0.1254902 ],
                            [1.        , 0.47843137, 0.36078431],
                            [0.95686275, 0.78431373, 0.        ],
                            [0.        , 0.3254902 , 0.54117647],
                            [0.50196078, 0.24313725, 0.45882353],
                            [0.96470588, 0.4627451 , 0.55686275],
                            [1.        , 0.40784314, 0.        ],
                            [0.57647059, 0.66666667, 0.        ],
                            [0.50588235, 0.43921569, 0.4       ],
                            [0.80784314, 0.63529412, 0.38431373],
                            [0.3254902 , 0.21568627, 0.47843137],
                            [0.70196078, 0.15686275, 0.31764706],
                            [1.        , 0.70196078, 0.        ],
                            [0.49803922, 0.09411765, 0.05098039],
                            [1.        , 0.55686275, 0.        ],
                            [0.34901961, 0.2       , 0.08235294],
                            [0.94509804, 0.22745098, 0.0745098 ],
                            [0.1372549 , 0.17254902, 0.08627451]])

UI_STRINGS = {
    "UI_BOT_NOSELECTED": "No hint/region is selected.",
    "UI_BOT_SELECTCOLOR": "Set the color of this group:",
    "UI_TOP_NOSELECTED": "No hint/region is selected.",
    "UI_TOP_SELECTLAYER": "Set the strength of this hint:",
    "UI_NONE": "(None)"
}
    
class UserPoint():

    def __init__(self, graph_coords: np.ndarray, index: list[int], color = np.ndarray, 
                 barycentric: list[float] = None, strength = 10, embedding = None, is_auto = False):
        """
            coords: location of clicked point in the sketch domain

            index: 3 vertices of triangle in which point lies

            group: color group 

            barycentric: barycentric coordinates of point within triangle

            strength: strength of point, default 10

            is_auto: specifies whether this point corresponds to a user hint or auto centroid
        """
        self.coords = graph_coords
        self.index = index
        self.barycentric = barycentric
        self.strength = strength
        self.embedding = embedding
        self.is_auto = is_auto
        
        # will intialize with a random Kelly color
        self.color = color

    def __repr__(self):
        return f"Point {self.coords} with color {self.color} and strength {self.strength}"
    
    def calc_embedding(self, features):
        if self.barycentric is None:
            raise ValueError("This point has no barycentric coordinates")
        self.embedding =    features[self.index[0]] * self.barycentric[0] + \
                            features[self.index[1]] * self.barycentric[1] + \
                            features[self.index[2]] * self.barycentric[2]
    
    # compatibility methods

    def set_color(self, c):
        self.color = c

    def set_strength(self, s):
        self.strength = s

    def get_plottable(self):
        return self.coords
        
class UserScribble():

    def __init__(self, pts: np.ndarray, mesh: TriMesh, color = np.ndarray, strength = 10):
        self.pts = pts
        self.uspts = []
        self.is_auto = False

        # create UserPoints on each triangle that this stroke passes through
        
        tri_avgs = np.zeros(mesh.tris.shape)
        tri_avg_counts = np.zeros(mesh.tris.shape[0])
        for pt in self.pts:
            tri_idx, bary = mesh.get_triangle(pt)
            tri_avgs[tri_idx] += np.array(bary)
            tri_avg_counts[tri_idx] += 1

        for tri in np.nonzero(tri_avg_counts)[0]:
            self.uspts.append(UserPoint(mesh.tri_centers[tri], mesh.tris[tri], color, tri_avgs[tri] / tri_avg_counts[tri], strength, None, False))

        # the following attributes overwrite whatever ones are stored in each UserPoint
        self.set_strength(strength)

        # will intialize with a random Kelly color
        self.set_color(color)

    def __repr__(self):
        return f"Scribble {self.uspts[0].coords} to {self.uspts[-1].coords} with color {self.color} and strength {self.strength}"
    
    def set_color(self, c):
        self.color = c
        for pt in self.uspts:
            pt.color = c

    def set_strength(self, s):
        self.strength = s
        for pt in self.uspts:
            pt.strength = s

    def get_plottable(self):
        return self.pts[:,0], self.pts[:,1]

class Plotter(FigureCanvasQTAgg):

    def __init__(self, parent):
        fig = Figure()
        self.ax = fig.add_subplot()
        super(Plotter, self).__init__(fig)
        self.setParent(parent)

        # axes
        self.lock_axis_scale = False
        self.current_axis_limits = [self.ax.get_xlim(), self.ax.get_ylim()]

        # sketch geometry
        self.sketch = None

        # interactive elements
        self.hints = []
        self.flat_hints = None
        self.selected = None
        self.picking_flag = False

        # for scribbles
        self.dt = 0
        self.dt_track = time.perf_counter()
        self.scribbling_event = False
        self.scribble_pts = []

        # for visualizing results
        self.labels = None
        self.mp_mesh = None
        self.mp_mesh_colors = None
        self.mp_boundary_poly_verts = None
        self.mp_boundary_poly_ids = None
        self.mp_boundary_poly_colors = None
        self.mp_boundary_poly_pc = None
        self.have_changed_labels = False # have any vertices be re-ID'd?
        self.have_changed_colors = False # have any colors been changed?

        self.smart_invert_axes(self.ax)
        
        # set up mouse input
        self.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.figure.canvas.mpl_connect('key_press_event', self.on_press)
        self.figure.canvas.mpl_connect('pick_event', self.on_pick)
        
        self.connect_axis_callbacks()
        self.update_window_controls()
    
    def connect_axis_callbacks(self):
        self.ax.callbacks.connect('xlim_changed', self.on_axis_resize)
        self.ax.callbacks.connect('ylim_changed', self.on_axis_resize)

    def update_window_controls(self):
        """

        Sets all UI elements (comboboxes, text, etc.) to their intended values based on program state
        Also toggles on/off certain window features depending on whether a point is selected
        
        """

        if self.sketch:
            self.window().group_auto.setEnabled(True)
            self.window().group_save.setEnabled(True)

            self.window().text_filename.setText(self.sketch.name)
            
        else:
            self.window().group_auto.setEnabled(False)
            self.window().group_save.setEnabled(False)

            self.window().text_filename.setText(UI_STRINGS['UI_NONE'])
        if self.selected:
            self.window().text_instruct.setText(UI_STRINGS['UI_TOP_SELECTLAYER'])
            self.window().text_setcolor.setText(UI_STRINGS['UI_BOT_SELECTCOLOR'])
            self.window().group_hintctrl.setEnabled(True)
            self.window().group_color.setEnabled(True)

            if self.selected.strength != self.window().slider_strength.value():
                self.window().slider_strength.setValue(self.selected.strength)

            # set color of "Pick color" box
            c = self.selected.color
            self.window().button_color.setStyleSheet('QPushButton {background-color: #' + 
                                                    ''.join([(lambda x: '0' + x if len(x) == 1 else x)(hex(val)[2:]) for val in np.round(c[:3] * 255).astype(np.int32)]) + 
                                                    '; color: black; border: none}')
        else:
            self.window().text_instruct.setText(UI_STRINGS['UI_TOP_NOSELECTED'])
            self.window().text_setcolor.setText(UI_STRINGS['UI_BOT_NOSELECTED'])
            self.window().group_hintctrl.setEnabled(False)
            self.window().group_color.setEnabled(False)

    def smart_invert_axes(self, ax):
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
    
    def on_axis_resize(self, event_ax):
        if not self.lock_axis_scale:
            self.current_axis_limits = [self.ax.get_xlim(), self.ax.get_ylim()]

    def on_press(self, event):
        pass

    def on_release(self, event):
        if self.figure.canvas.toolbar.mode == '' and not self.sketch is None and self.scribbling_event:
            self.scribbling_event = False

            if len(self.scribble_pts) > 2:
                clicked = self.add_clicked(np.array(self.scribble_pts))
                print(f"Added scribble {clicked}, which has {clicked.pts.shape[0]} points and passes through {len(clicked.uspts)} triangles")
            else:
                clicked = self.add_clicked(np.array(self.scribble_pts[0]))
            self.selected = clicked
            self.have_changed_labels = True
            self.redraw()
                
    
    def on_mouse_move(self, event):
        THRESH = 0.015
        if self.scribbling_event:
            this_time = time.perf_counter()
            this_delta = this_time - self.dt_track
            self.dt_track = this_time
            if self.dt + this_delta >= THRESH:
                self.dt = 0
                self.scribble_pts.append((event.xdata, event.ydata))
                self.ax.plot([self.scribble_pts[-2][0], self.scribble_pts[-1][0]], [self.scribble_pts[-2][1], self.scribble_pts[-1][1]], color='b')
                self.draw()
            else:
                self.dt += this_delta

    def on_click(self, event):
        """ CLICK SHOULD NEVER REDRAW. Only pick or release """
        if event.xdata is None:
            return
        if self.figure.canvas.toolbar.mode == '' and not self.sketch is None:

            if self.picking_flag:
                self.picking_flag = False
            elif (self.sketch.bl.x <= event.xdata <= self.sketch.br.x) and (self.sketch.br.y <= event.ydata <= self.sketch.tr.y):
                self.scribbling_event = True
                self.scribble_pts = [(event.xdata, event.ydata)]
            
            
        
    def on_scroll(self, event):
        if event.button == 'down':
            self.window().slider_strength.setValue(self.window().slider_strength.value() - 1)
        else:
            self.window().slider_strength.setValue(self.window().slider_strength.value() + 1)

    def on_pick(self, event):
        self.picking_flag = True
        self.selected = event.artist.obj
        print(f"Clicked on previous point {event.artist.obj}")
        self.redraw()

    def reset_plot_attributes(self, sketch=False):
        print("Clearing")
        if sketch:
            self.sketch = None
        self.hints = []
        self.selected = None
        self.mp_boundary_poly_colors = None
        self.mp_boundary_poly_ids = None
        self.mp_boundary_poly_pc = None
        self.mp_boundary_poly_verts = None
        self.mp_mesh_colors = None
        self.ax.cla()
        self.connect_axis_callbacks()

        self.update_window_controls()
        self.draw()

    def calc_labels(self):
        """
        
            updates the following attributes:
            -   self.labels
            -   self.mp_boundary_poly_verts
            -   self.mp_boundary_poly_ids
            -   self.mp_boundary_poly_pc
            
        """

        centroids, centroid_powers = self.compute_centroids_from_hints()
        labels_pre = self.compute_clusters(centroids, centroid_powers)
        centroids, centroid_powers = self.flatten_scribbles(labels_pre) # operates on labels_pre in place
        self.labels, centroids = self.process_clusters(labels_pre, centroids)
        t1 = time.perf_counter()
        self.mp_boundary_poly_verts, self.mp_boundary_poly_ids = self.generate_contour_polygons(self.labels, labels_pre, centroids, centroid_powers)
        print(f"Finished boundary extraction, took {time.perf_counter() - t1} s")
        t1 = time.perf_counter()
        self.mp_boundary_poly_pc = PolyCollection(self.mp_boundary_poly_verts)
        print(f"Rebuilt polygons in {time.perf_counter() - t1} s")

    def calc_colors(self):
        """
        
            updates the following attributes:
            -   self.mp_boundary_poly_colors
            -   self.mp_boundary_poly_pc
            -   self.mp_mesh_colors
            -   self.mp_mesh
        
        """
        t1 = time.perf_counter()
        if len(self.mp_boundary_poly_ids) > 0:
            self.mp_boundary_poly_colors = self.gen_color_table(self.mp_boundary_poly_ids)
            self.mp_boundary_poly_pc.set_facecolor(self.mp_boundary_poly_colors)
        else:
            self.mp_boundary_poly_pc = None

        self.mp_mesh_colors = self.gen_color_table(self.labels)
        self.mp_mesh.set_facecolor(self.mp_mesh_colors)
        print(f"Recomputed face colors in {time.perf_counter() - t1} s")
        

    def redraw(self):
        """

            mega function for redrawing the canvas whenever the program's state changes

            helpers (to keep it simple, nothing else should call these, generally)
                -   update_window_controls()
                -   calc_labels()
                -   calc_colors()
            
        """

        self.ax.cla()
        self.connect_axis_callbacks()

        self.lock_axis_scale = True


        if self.hints:

            # update label & color data if hints have changed
            if self.have_changed_labels:
                self.calc_labels()
                self.calc_colors()
            elif self.have_changed_colors:
                self.calc_colors()
            
            t1 = time.perf_counter()
            # draw regions

            if not self.mp_mesh is None:
                self.ax.add_collection(self.mp_mesh)

            if not self.mp_boundary_poly_pc is None:
                pass
                self.ax.add_collection(self.mp_boundary_poly_pc)
        else:
            t1 = time.perf_counter()

        # draw strokes
        
        self.draw_sketch(self.sketch, self.ax)
            
        # draw points

        self.plot_pts(self.ax)

        # restore axes

        self.ax.axis('equal')
        self.smart_invert_axes(self.ax)

        self.ax.set_xlim(self.current_axis_limits[0])
        self.ax.set_ylim(self.current_axis_limits[1])
        self.ax.axis('off')

        self.lock_axis_scale = False
        self.update_window_controls()
        self.draw()
        print(f"Redrew canvas in {time.perf_counter() - t1} s")
    
    def gen_color_table(self, labels: np.ndarray):

        colors = np.zeros((len(labels), 3), dtype=np.float64)

        colors[labels == -1] = np.array([1.,1.,1.])
        for idx in range(np.max(labels) + 1):
            colors[labels == idx] = self.hints[idx].color
        return colors

    def draw_sketch(self, sketch: Sketch, ax: plt.Axes):

        # draw curve & direction of dipole for each edge
        for stroke in sketch.strokes:
            x, y = stroke.get_edges()
            ax.plot(x,y,marker="None",color="black",solid_capstyle='round',linewidth=2,zorder=1)
    
    def load_sketch(self, filepath):
        self.reset_plot_attributes(sketch=True)
        self.sketch = Sketch(filepath)
        self.features = self.sketch.features
        self.verts = self.sketch.trimesh.verts
        self.tris = self.sketch.trimesh.tris
        self.mp_mesh = TM(self.sketch.triangulation)

        # sets axis limits correctly
        self.draw_sketch(self.sketch, self.ax)
        self.ax.axis('equal')
        self.smart_invert_axes(self.ax) 

        self.redraw()
        
    def on_update_slider(self, idx):
        if not (self.selected is None):
            self.selected.set_strength(idx)
            self.have_changed_labels = True
            self.redraw()

    def pick_color(self):
        c = QColorDialog.getColor()
        if (c.isValid()) and not (self.selected is None):
            self.selected.color = np.array([val / 255 for val in c.getRgb()][:3])
        self.have_changed_colors = True
        self.redraw()

    def pick_random_color(self):
        """ 
        
            Tries to select from KELLY_COLORS
            
        """
        
        kelly_used_mask = np.zeros(KELLY_COLORS.shape[0])
        for hint in self.hints:
            find_kelly = np.where(np.apply_along_axis(np.all, 1, hint.color == KELLY_COLORS))[0]
            if find_kelly.size > 0:
                kelly_used_mask[find_kelly[0]] = 1

        avail_kelly = np.where(kelly_used_mask == 0)[0]
        if avail_kelly.size > 0:
            return KELLY_COLORS[avail_kelly[0]]
        else:
            return np.random.rand(3)

    
    def add_clicked(self, point: np.ndarray, group=None):

        # check if this point is inside each triangle
        
        trimesh = self.sketch.trimesh
        color = self.pick_random_color()
            
        if point.ndim == 1:
            tri_idx, bary = trimesh.get_triangle(point)
            in_tri = trimesh.tris[tri_idx]
            
            print(f"New point is in triangle {in_tri} at {trimesh.tri_centers[tri_idx]}\n\tBary: {bary}")

            pt = UserPoint(point, in_tri, color, bary)

            self.hints.append(pt)
            return pt
        elif point.ndim == 2:
            # scribble
            scrib = UserScribble(point, trimesh, color)
            self.hints.append(scrib)
            return scrib
        
    def flatten_scribbles(self, labels: np.ndarray):
        """
            Replaces cluster ID's with hint ID's,
            consolidating labels that belong to the same hint
            (Basically: a fix for scribbles)
        """
        counter = 0
        centroid_powers = np.zeros(len(self.hints))
        for idx, hint in enumerate(self.hints):
            if isinstance(hint, UserPoint):
                labels[labels == counter] = idx
                counter += 1
            elif isinstance(hint, UserScribble):
                labels[np.logical_and(labels >= counter, labels < counter + len(hint.uspts))] = idx
                counter += len(hint.uspts)
            centroid_powers[idx] = hint.strength

        # recompute centroids
        features = self.features
        no_centroids = np.max(labels) + 1
        centroids = np.zeros((no_centroids, features.shape[1]), dtype=np.float64)
        for label in range(no_centroids):
            sel = labels == label
            try:
                centroids[label] = np.average(features[sel], axis=0, weights=self.sketch.trimesh.vertex_weights[sel])
            except ZeroDivisionError as e:
                # this cluster has no vertices
                if centroids[label][0] != np.inf: print(f"WARNING: Cluster {label} has no vertices")
                centroids[label][0] = np.inf


        return centroids, centroid_powers


    def process_clusters(self, lbls_pre, centroids):
        # handle disconnected components

        print("Postprocessing clusters by splitting")
        t1 = time.perf_counter()

        labels = deepcopy(lbls_pre)
        verts = self.verts
        tris = self.tris
        features = self.features
        nodes = self.sketch.trimesh.nodes
        cluster_families = {}

        for c in range(centroids.shape[0]):
            indices = np.argwhere(labels == c).flatten()
            this_hint = self.hints[c]
            # add priviledged "first index" of whatver vertex is closest to this centroid
            if isinstance(this_hint, UserPoint):
                indices = np.insert(indices, 0, this_hint.index[np.argmax(this_hint.barycentric)])
            elif isinstance(this_hint, UserScribble):
                indices = np.insert(indices, 0, this_hint.uspts[0].index[np.argmax(this_hint.uspts[0].barycentric)])
            # run a BFS from the first node in the cluster,
            # where only nodes within the cluster are discoverable.
            # if not all indices are covered,
            # run another BFS to create a new cluster, etc.
            clusters = []
            all_discovered_table = np.zeros(len(nodes), dtype=np.int32)
            for idx in indices:
                if all_discovered_table[idx]:
                    continue
                queue = [idx]
                discovered = [idx]
                discovered_table = np.zeros(len(nodes), dtype=np.int32)
                discovered_table[idx] = 1
                all_discovered_table[idx] = 1
                while (len(queue) != 0):
                    current = queue.pop(0)
                    for neigh in nodes[current].neighs:
                        if (not ((neigh is None) or (discovered_table[neigh.global_id]))) and (labels[neigh.global_id] == c):
                            discovered.append(neigh.global_id)
                            discovered_table[neigh.global_id] = 1
                            all_discovered_table[neigh.global_id] = 1
                            queue.append(neigh.global_id)
                clusters.append(discovered)
            if len(clusters) > 1:
                cluster_family = [c]
                for cluster in clusters[1:]:
                    cluster_family.append(np.max(labels) + 1)
                    labels[cluster] = np.max(labels) + 1
                for c in cluster_family:
                    cluster_families[c] = cluster_family
        print(f"Finished cluster splitting, took {time.perf_counter() - t1} seconds")

        """
        if any additional clusters were born: merge ones not connected to each centroid
            iterate over all triangles, looking for mixed triangles
            for each pair of clusters represented in at least one mixed triangle,
            merge into the cluster with the largest cumulative length of shared edges
        """
       

        t2 = time.perf_counter()

        print("Starting cluster merge")
        while True:
            updated = False
            shared_edge = np.zeros((np.max(labels) + 1, np.max(labels) + 1), dtype=np.float64)
            cluster_areas = np.zeros((np.max(labels) + 1), dtype=np.float64)
            # recalculate mixed triangles and cluster areas
            for i, tri in enumerate(tris):
                match_01 = (labels[tri[0]] == labels[tri[1]])
                match_12 = (labels[tri[1]] == labels[tri[2]])
                match_02 = (labels[tri[0]] == labels[tri[2]])
                if match_01 and match_12 and match_02:
                    # triangle is entirely within a cluster
                    cluster_areas[labels[tri[0]]] += self.sketch.trimesh.tri_areas[i]
                elif match_01 or match_12 or match_02:
                    # simple mixed triangle
                    if match_01:
                        m1 = tri[0]
                        m2 = tri[1]
                        other = tri[2]
                    elif match_12:
                        m1 = tri[1]
                        m2 = tri[2]
                        other = tri[0]
                    else:
                        m1 = tri[0]
                        m2 = tri[2]
                        other = tri[1]
                    id1, id2 = min(labels[m1], labels[other]), max(labels[m1], labels[other])
                    shared_edge[id1][id2] += np.sqrt(np.sum((verts[m1] - verts[m2]) ** 2)) * 0.5
                else:
                    # complex mixed triangle
                    vert_ids = np.array([tri[0], tri[1], tri[2]], dtype=np.int32)
                    vert_labels = np.array([labels[tri[0]], labels[tri[1]], labels[tri[2]]], dtype=np.int32)
                    order = np.argsort(vert_labels)
                    vert_ids = np.array(vert_ids)[order]
                    vert_labels = np.array(vert_labels)[order]
                    # handle each of three edges
                    edge_len = np.sqrt(np.sum((verts[vert_ids[1]] - verts[vert_ids[2]]) ** 2)) * 0.5
                    shared_edge[vert_labels[0]][vert_labels[1]] += edge_len
                    shared_edge[vert_labels[0]][vert_labels[2]] += edge_len
                    edge_len = np.sqrt(np.sum((verts[vert_ids[0]] - verts[vert_ids[2]]) ** 2)) * 0.5
                    shared_edge[vert_labels[0]][vert_labels[1]] += edge_len
                    shared_edge[vert_labels[1]][vert_labels[2]] += edge_len
                    edge_len = np.sqrt(np.sum((verts[vert_ids[0]] - verts[vert_ids[1]]) ** 2)) * 0.5
                    shared_edge[vert_labels[0]][vert_labels[2]] += edge_len
                    shared_edge[vert_labels[1]][vert_labels[2]] += edge_len
            # merge clusters
            cluster_pairs = np.vstack(np.nonzero(shared_edge)).T # n_clusters x 2
            for c_ in cluster_families:
                c_fam = cluster_families[c_]
                if len(c_fam) == 1:
                    continue
                areas = [cluster_areas[c] for c in c_fam]
                c_fam_sorted = np.append(np.array(c_fam[1:])[np.flip(np.argsort(areas[1:]))], c_fam[0])

                # now, merge largest "non-canon" cluster
                # with whatever neighbor has largest shared perimeter
                
                smallest_c = c_fam_sorted[0]
                candidates = cluster_pairs[cluster_pairs[:,0] == smallest_c][:,1]
                candidates = np.hstack((candidates, cluster_pairs[cluster_pairs[:,1] == smallest_c][:,0]))

                if len(candidates) == 0:
                    # it's impossible to merge this cluster because it has no neighbors
                    updated = True
                    c_fam.remove(smallest_c)
                    cluster_families.pop(smallest_c)
                    for c__ in c_fam:
                        cluster_families[c__] = c_fam
                    labels[labels == smallest_c] = -1
                    break

                new_c = candidates[np.argsort([cluster_areas[c] for c in candidates])[-1]]

                # merge
                updated = True
                labels[labels == smallest_c] = new_c
                c_fam.remove(smallest_c)
                cluster_families.pop(smallest_c)
                for c__ in c_fam:
                    cluster_families[c__] = c_fam

                break # this is slow!!!
            
            if not updated:
                break

        # finally, compute new centroids for each cluster
        
        centroids = np.zeros(centroids.shape, dtype=np.float64)
        for label in range(centroids.shape[0]):
            sel = labels == label
            try:
                centroids[label] = np.average(features[sel], axis=0, weights=self.sketch.trimesh.vertex_weights[sel])
            except ZeroDivisionError as e:
                # this cluster has no vertices
                if centroids[label][0] != np.inf: print(f"WARNING: Cluster {label} has no vertices")
                centroids[label][0] = np.inf
    
        print(f"Finished cluster merge, took {time.perf_counter() - t2} seconds")
        print(f"Finished post-processing, took {time.perf_counter() - t1} seconds")
        return labels, centroids
    
    def find_halfspace_intersect(self, p1, p2, c1, c2, r1, r2) -> float:
        """
            returns the interpolant, not the point
        """
        d = np.linalg.norm(c1 - c2)
        t = ((pow(d, 2) + r1 - r2) / (2 * d))
        s = t * ((c1 - c2) / d) + (d - t) * ((c2 - c1) / d)
        num = ((np.dot(c1, c1) - np.dot(c2, c2) + np.dot(s, c2) - np.dot(s, c1)) / 2) - np.dot(p1, c1) + np.dot(p1, c2)
        denom = np.dot(p2 - p1, c1 - c2)
        interp = num / denom
        return interp

    def generate_contour_polygons(self, labels, labels_og, centroids, centroids_strength = None):

        features = self.features

        polys = []
        polys_labels = []

        debug_edges = []
        debug_verts = []

        mixed_edge_splits = np.zeros([self.sketch.trimesh.nodes.shape[0], self.sketch.trimesh.nodes.shape[0], 2], dtype=np.float64)

        for e in self.sketch.trimesh.my_edges:
            # for each mixed edge,
            # estimate the point where it crosses the boundary between Voronoi cells

            if (labels[e[0]] == labels[e[1]]):
                continue
            else:
                p1, p2 = min(e[0], e[1]), max(e[0], e[1])

                # use the ORIGINAL labels here
                label1, label2 = labels_og[p1], labels_og[p2]

                if (label1 == label2):
                    raise ValueError(f"Unable to split {p1} and {p2}: same original cluster")

                if centroids_strength is None:
                    r1 = 0
                    r2 = 0
                else:
                    r1 = centroids_strength[label1] * self.sketch.power_inc
                    r2 = centroids_strength[label2] * self.sketch.power_inc

                interp = self.find_halfspace_intersect(features[p1], features[p2], centroids[label1], centroids[label2], r1, r2)
                if not (0 <= interp <= 1): interp = np.clip(interp, 0, 1)
                mixed_edge_splits[p1][p2] = self.sketch.trimesh.nodes[p1].pos.coords + interp * (self.sketch.trimesh.nodes[p2].pos.coords - self.sketch.trimesh.nodes[p1].pos.coords)

        for i, tri in enumerate(self.sketch.trimesh.tris):
            match_01 = (labels[tri[0]] == labels[tri[1]])
            match_12 = (labels[tri[1]] == labels[tri[2]])
            match_02 = (labels[tri[0]] == labels[tri[2]])
            if match_01 and match_12 and match_02:
                # triangle is entirely within a cluster
                continue
            elif match_01 or match_12 or match_02:
                # simple mixed triangle
                if match_01:
                    m1 = min(tri[0], tri[1])
                    m2 = max(tri[0], tri[1])
                    other = tri[2]
                elif match_12:
                    m1 = min(tri[1], tri[2])
                    m2 = max(tri[1], tri[2])
                    other = tri[0]
                else:
                    m1 = min(tri[0], tri[2])
                    m2 = max(tri[0], tri[2])
                    other = tri[1]
                # create quad. and triangle
                quad_v = np.array([  mixed_edge_splits[min(m1, other)][max(m1, other)],
                            mixed_edge_splits[min(m2, other)][max(m2, other)],
                            self.sketch.trimesh.nodes[m2].pos.coords,
                            self.sketch.trimesh.nodes[m1].pos.coords
                            ])
                tri_v = np.array([   mixed_edge_splits[min(m1, other)][max(m1, other)],
                            mixed_edge_splits[min(m2, other)][max(m2, other)],
                            self.sketch.trimesh.nodes[other].pos.coords
                            ])
                
                polys.append(quad_v)
                polys_labels.append(labels[m1])
                polys.append(tri_v)
                polys_labels.append(labels[other])
            else:
                # complex mixed triangle
                # DIFFERENT TO PAPER: use triangle barycenter all the time, instead of as a fallback
                # This is less error prone than the previous "intersection of halfspaces" approach
                # And is negligibly different in terms of the quality of the results

                center = self.sketch.trimesh.tri_centers[i]
    
                # create 3 quads
                quad1_v = np.array([ center,
                            mixed_edge_splits[min(tri[0], tri[1])][max(tri[0], tri[1])],
                            self.sketch.trimesh.nodes[tri[0]].pos.coords,
                            mixed_edge_splits[min(tri[0], tri[2])][max(tri[0], tri[2])]
                           ])
                quad2_v = np.array([ center,
                            mixed_edge_splits[min(tri[1], tri[0])][max(tri[1], tri[0])],
                            self.sketch.trimesh.nodes[tri[1]].pos.coords,
                            mixed_edge_splits[min(tri[1], tri[2])][max(tri[1], tri[2])]
                           ])
                quad3_v = np.array([ center,
                            mixed_edge_splits[min(tri[2], tri[0])][max(tri[2], tri[0])],
                            self.sketch.trimesh.nodes[tri[2]].pos.coords,
                            mixed_edge_splits[min(tri[2], tri[1])][max(tri[2], tri[1])]
                           ])
                
                polys.append(quad1_v)
                polys_labels.append(labels[tri[0]])
                polys.append(quad2_v)
                polys_labels.append(labels[tri[1]])
                polys.append(quad3_v)
                polys_labels.append(labels[tri[2]])

        return polys, np.array(polys_labels)

    def find_closest_nodes(self, centroids):
        """

            Map auto-generated centroids to vertices on the mesh

        """
        dists = np.zeros((centroids.shape[0], self.features.shape[0]))
        for i, centroid in enumerate(centroids):
            difference = self.features - centroid
            dists[i] = np.multiply(difference, difference).sum(1)
        return np.argmin(dists, axis=1)
    
    def compute_clusters(self, centroids, centroid_powers = None):
        dists = np.zeros((centroids.shape[0], self.features.shape[0]))
        for i, centroid in enumerate(centroids):
            difference = self.features - centroid
            if not (centroid_powers is None):
                dists[i] = np.multiply(difference, difference).sum(1) - (centroid_powers[i] * self.sketch.power_inc)
            else:
                dists[i] = np.multiply(difference, difference).sum(1)
        return np.argmin(dists.T, axis=1)
    
    def compute_centroids_from_hints(self):

        self.flat_hints = []

        for pt in self.hints:
            if isinstance(pt, UserPoint):
                if pt.embedding is None:
                    pt.calc_embedding(self.features)
                self.flat_hints.append(pt)
            elif isinstance(pt, UserScribble):
                for _pt in pt.uspts:
                    if _pt.embedding is None:
                        _pt.calc_embedding(self.features)
                    self.flat_hints.append(_pt)

        centroids = []
        centroid_powers = []
        
        for pt in self.flat_hints:
            centroids.append(pt.embedding)
            centroid_powers.append(pt.strength)
        
        return np.array(centroids), np.array(centroid_powers)
    
    def gen_auto_seeds(self):

        self.reset_plot_attributes()

        init_method = 'area-weighted'
        weights = self.sketch.trimesh.vertex_weights
        features = self.features
        model = KMeans(self.sketch, n_clusters=self.window().spin_regions.value(), weights=weights, init_method=init_method, n_init=1, USE_LOG=True)
        
        print("Clustering...")
        t3 = time.perf_counter()

        # get clusters out
        labels, centroids, centroids_init, centroids_init_nodes, all_res, error, discovery_maps = model.fit(features)
        closest = self.find_closest_nodes(centroids)

        self.hints = []
        for i, v_idx in enumerate(closest):
            color = self.pick_random_color()
            pt = UserPoint(self.verts[v_idx], [v_idx, -1, -1], color, [1, 0, 0], is_auto=True, embedding=centroids[i])
            self.hints.append(pt)
        self.have_changed_labels = True

        print(f"Finished clustering on {features.shape[1]} features, took {time.perf_counter() - t3} seconds")

        # sets axis limits correctly
        self.draw_sketch(self.sketch, self.ax)
        self.ax.axis('equal')
        self.smart_invert_axes(self.ax) 

        self.redraw()

    def save_png(self):

        save_strokes = self.window().check_plot_strokes.isChecked()
        save_regions = self.window().check_plot_regions.isChecked()
        save_auto = self.window().check_plot_auto.isChecked()
        save_user = self.window().check_plot_user.isChecked()

        fig_save = plt.figure()
        ax_save = fig_save.add_subplot()

        if save_regions:
            if not self.mp_mesh is None:
                mesh = TM(self.sketch.triangulation)
                mesh.set_facecolor(self.mp_mesh_colors)
                ax_save.add_collection(mesh)
            if not self.mp_boundary_poly_pc is None:
                pc = PolyCollection(self.mp_boundary_poly_verts)
                pc.set_facecolor(self.mp_boundary_poly_colors)
                ax_save.add_collection(pc)
        if save_strokes:
            self.draw_sketch(self.sketch, ax_save)
        
        self.plot_pts(ax_save, final=True, auto=save_auto, user=save_user)
        
        ax_save.axis('equal')
        ax_save.set_xlim([self.sketch.xmin - 50, self.sketch.xmax + 50])
        ax_save.set_ylim([self.sketch.ymin - 50, self.sketch.ymax + 50])
        ax_save.invert_yaxis()
        ax_save.axis('off')
        fig_save.savefig('.'.join(self.sketch.path.split('.')[:-1])+ '_out.png', bbox_inches='tight', format='png', dpi=600)

        
    def plot_pts(self, ax, final=False, auto=True, user=True):

        strength_to_txt_label = lambda x : ('+' if x - 10 > 0 else '') + str(x - 10)

        for i,pt in enumerate(self.hints):

            if pt.is_auto:
                if i == 0:
                    continue
                if not auto:
                    continue
            else:
                if not user:
                    continue

            if final:
                color = pt.color
            else:
                if isinstance(pt, UserPoint):
                    if pt == self.selected:
                        color = 'g'
                    else:
                        if pt.is_auto:
                            color = 'grey'
                        else:
                            color = 'white'
                else:
                    # scribble
                    if pt == self.selected:
                        color = 'b'
                    else:
                        color = 'grey'
                    
            if isinstance(pt, UserPoint):
                artist = ax.scatter(*pt.coords, marker='o', color=color, edgecolors='black', picker=True)
            else:
                artist = ax.plot(*pt.get_plottable(), color=color, picker=True)[0]
            artist.obj = pt

            if final:
                if isinstance(pt, UserScribble):
                    x, y = pt.pts[0]
                else:
                    x, y = pt.coords

                if pt.strength != 10: # if it has been changed from the default by the user
                    ax.text(x + 3, y - 3, strength_to_txt_label(pt.strength))
                elif pt.is_auto:
                    ax.text(x - 4, y + 4, "a", size='x-small')

        
