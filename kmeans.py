"""
    Implementation of K-Means with Lloyd's Algorithm
    Also, other clustering stuff, like auto-initialization of seed points
"""

import numpy as np
import math
from collections import namedtuple
from sklearn.cluster import kmeans_plusplus
import time

KMeans_res = namedtuple("KMeans_res", "labels centroids_final centroids_og centroids_labels")

def assess_error(data, centroids, labels):
    total = 0.
    num_classes = np.max(labels) + 1 # assuming labels start from 0
    for label in range(num_classes):
        sel = labels == label
        total += np.sum(np.power(data[sel] - centroids[label], 2))
    return total

class KMeans():

    def __init__(self, sketch, n_clusters = 8, init = None, n_init = 10, max_iter: int = 300, tol: float = 0.00001, weights=None, init_method='default',
                 DISTANCE_COEFF=1, AREA_COEFF=1, USE_LOG=False):
        """
            init:   np.ndarray if specifying initial center guesses
        """
        if isinstance(init, np.ndarray):
            if len(init.shape) != 2: raise ValueError("Must init. with 2D array of centroids")
        else: 
            init = None

        self.sketch = sketch
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.centroids_final = None
        self.labels = None
        self.weights = weights
        self.n_init = n_init
        self.init_method = init_method

        # a little bit of caching to speed up the initialization and clustering
        # key:      id of a triangle
        # value:    array of len (tris) containing squared distances from this face
        #           to all tri centers
        self.paired_distances_dict = {}
        # key:      id of a centroid (uniquely identifies centroids that are calculated 
        #           over the course of the algorithm)
        # value:    array of len (vertices) containing squared distances from this face
        #           to all vertices
        self.paired_v_distances_dict = {}

        self._centroid_id_counter = 0
        
        self.DISTANCE_COEFF = DISTANCE_COEFF
        self.AREA_COEFF = AREA_COEFF
        self.USE_LOG = USE_LOG

    @property
    def cluster_centers_(self):
        return self.centroids_final

    @staticmethod
    def distance(point: np.ndarray, centroids: np.ndarray):
        """
            compute squared distances from a feature vector to all centroids
        """
        return np.multiply(centroids - point, centroids - point).sum(1)
    
    def find_closest_cluster(self, data, centroids, unique_ctrd_identifiers):
        """
            return the label (index) of the cluster whose centroid is closest
        """
        dists = []
        for j in range(centroids.shape[0]):
            if not unique_ctrd_identifiers[j] in self.paired_v_distances_dict:
                self.paired_v_distances_dict[unique_ctrd_identifiers[j]] = self.distance(centroids[j], data)
            dists.append(self.paired_v_distances_dict[unique_ctrd_identifiers[j]])             
        dists = np.array(dists)
        res = np.apply_along_axis(np.argsort, 0, dists)
        return res[0]
    
    def compute_centroids(self, data: np.ndarray, labels):
        """
            re-compute centroids from data/labels
        """
        num_classes = np.max(labels) + 1 # assuming labels start from 0
        centroids = np.zeros((num_classes, data.shape[1]), dtype=np.float64)
        for label in range(num_classes):
            sel = labels == label
            try:
                centroids[label] = np.average(data[sel], axis=0, weights=(self.weights[sel] if not self.weights is None else None))
            except ZeroDivisionError as e:
                # this cluster has no vertices anymore
                if centroids[label][0] != np.inf: print(f"WARNING: Cluster {label} has no more vertices left")
                centroids[label][0] = np.inf
        return centroids
    
    def pick_new_centroid(self, data, centroids_current, centroids_labels, count, area_prob):
        """
            pick a new centroid based on the current choices
        """

        dists = []
        for j in range(count):
            if not centroids_labels[j] in self.paired_distances_dict:
                self.paired_distances_dict[centroids_labels[j]] = self.distance(centroids_current[j], data)
            dists.append(self.paired_distances_dict[centroids_labels[j]])             
        dists = np.array(dists)
        min_dists = np.min(dists, axis=0).flatten()
        min_dists *= self.DISTANCE_COEFF
        if area_prob is None:
            prob = min_dists
        else:
            prob = area_prob
            prob = np.multiply(min_dists, area_prob)

        # trimesh = self.sketch.trimesh
        # STD_DEV = 1000
        # mean_diff = (trimesh.tri_centers - self.sketch.center).sum(1) / STD_DEV
        # prob = np.multiply(prob, np.exp(-(np.multiply(mean_diff, mean_diff))))

        idx_chosen = np.random.choice(data.shape[0], size=1, p=prob/np.sum(prob))[0]
        return idx_chosen, min_dists, prob
    
    def _next_candidate(self, tri_area_order, tris_covered, tris_picked):
        for idx in tri_area_order:
            if not (tris_covered[idx] or tris_picked[idx]):
                return idx
        return None

    def pick_new_centroid_det(self, data, centroids_current, centroids_labels, count, tri_areas, bg_mask):

        dists = []
        for j in range(count):
            if not centroids_labels[j] in self.paired_distances_dict:
                self.paired_distances_dict[centroids_labels[j]] = self.distance(centroids_current[j], data)
            dists.append(self.paired_distances_dict[centroids_labels[j]])             
        dists = np.array(dists)
        min_dists = np.min(dists, axis=0).flatten()
        min_dists *= self.DISTANCE_COEFF
        
        prob = np.multiply(min_dists, tri_areas)
        prob = np.multiply(prob, 1 - bg_mask)

        tri_order = np.flip(np.argsort(prob))

        idx_chosen = tri_order[0]
        return idx_chosen, min_dists, prob
    
    def update_coverage(self, next, tris_covered, edge_criterion):
        trimesh = self.sketch.trimesh
        tri_graph = trimesh.tri_graph
        bfs_discovered = np.zeros(trimesh.tris.shape[0])
        queue = [next]
        el_thresh = edge_criterion[next] / 3
        while queue:
            current = queue.pop(0)
            for n_i, neigh in enumerate(tri_graph[current].neighs):
                if not bfs_discovered[neigh]:
                    # see if this neighbor is really "connected"
                    # according to our edge length criterion
                    if tri_graph[current].weights[n_i] >= el_thresh:
                        bfs_discovered[neigh] = 1
                        tris_covered[neigh] = 1
                        queue.append(neigh)
                    else:
                        pass
        return bfs_discovered

    def get_largest_corner_tris(self):

        trimesh = self.sketch.trimesh
        tri_areas = trimesh.tri_areas
        
        ctri_all = None
        ctri_tr = None
        ctri_tl = None
        ctri_br = None
        ctri_bl = None
        ctri_all_a = -1
        ctri_tr_a = -1
        ctri_tl_a = -1
        ctri_br_a = -1
        ctri_bl_a = -1

        for tri in (trimesh.tl.tris):
            if tri_areas[tri] > ctri_all_a:
                ctri_all_a = tri_areas[tri]
                ctri_all = tri
                ctri_tl_a = tri_areas[tri]
                ctri_tl = tri
            elif tri_areas[tri] > ctri_tl_a:
                ctri_tl_a = tri_areas[tri]
                ctri_tl = tri
        
        for tri in (trimesh.tr.tris):
            if tri_areas[tri] > ctri_all_a:
                ctri_all_a = tri_areas[tri]
                ctri_all = tri
                ctri_tr_a = tri_areas[tri]
                ctri_tr = tri
            elif tri_areas[tri] > ctri_tr_a:
                ctri_tr_a = tri_areas[tri]
                ctri_tr = tri

        for tri in (trimesh.bl.tris):
            if tri_areas[tri] > ctri_all_a:
                ctri_all_a = tri_areas[tri]
                ctri_all = tri
                ctri_bl_a = tri_areas[tri]
                ctri_bl = tri
            elif tri_areas[tri] > ctri_bl_a:
                ctri_bl_a = tri_areas[tri]
                ctri_bl = tri

        for tri in (trimesh.br.tris):
            if tri_areas[tri] > ctri_all_a:
                ctri_all_a = tri_areas[tri]
                ctri_all = tri
                ctri_br_a = tri_areas[tri]
                ctri_br = tri
            elif tri_areas[tri] > ctri_br_a:
                ctri_br_a = tri_areas[tri]
                ctri_br = tri

        return ctri_all, ctri_tl, ctri_tr, ctri_bl, ctri_br
    
    def init_coverage_map_det(self, tri_areas, edge_criterion):
        
        trimesh = self.sketch.trimesh
        maps = []
        tris_picked = []
        bg_mask = np.zeros(len(trimesh.tris))
        tri_area_order = np.flip(np.argsort(tri_areas))
        total_a = (trimesh.tr.pos.y - trimesh.br.pos.y) * (trimesh.br.pos.x - trimesh.bl.pos.x)

        i = 0
        while (True):
            disc_map_curr = np.zeros(len(trimesh.tris))
            new_seed = tri_area_order[i]
            self.update_coverage(new_seed, disc_map_curr, edge_criterion)

            maps.append(disc_map_curr)
            tris_picked.append(new_seed)

            if (disc_map_curr).all():
                # all triangles filled. not good!
                print("\tWARNING: BG mask initialization failed, all triangles covered")
                return -1, None
            
            
            new_area = 0
            for j in range(len(trimesh.tris)):
                if disc_map_curr[j]:
                    new_area += tri_areas[j]
            if new_area / total_a >= 0.975:
                # we've covered too much of this sketch by area for this map to be useful
                # fall back
                print(f"\tWARNING: BG mask initialization failed, too much area covered")
                return -1, None
            
            # check if superset of a previous discovery map
            if (disc_map_curr - bg_mask >= 0).all():
                bg_mask = disc_map_curr
            else:
                # time to stop
                break

            i += 1

        return tri_area_order[0], bg_mask 
    
    def init_centroids_deterministic(self, data):
        
        trimesh = self.sketch.trimesh
        tri_areas = trimesh.tri_areas

        centroids = np.zeros((self.n_clusters, data.shape[1]), dtype=np.float64)
        centroids_labels = []
        discovery_maps = []

        if self.USE_LOG:
            print(f"\tUsing sqrt area weights")
            area_prob = np.sqrt(self.sketch.trimesh.tri_areas)
        else:
            print(f"\tUsing area weights (unmodified)")
            area_prob = self.sketch.trimesh.tri_areas
        area_prob *= self.AREA_COEFF

        # calculate coordinates for each triangular face in the embedding space
        # also, calculate average per-tri edge length

        data_tris = np.zeros((trimesh.tris.shape[0], data.shape[1]), dtype=np.float64)
        el_avg_tris = np.zeros(trimesh.tris.shape[0], dtype=np.float64)
        eq_edge_tris = np.zeros(trimesh.tris.shape[0], dtype=np.float64)
        for i, tri in enumerate(trimesh.tris):
            eq_edge_tris[i] = math.sqrt((4 / math.sqrt(3)) * tri_areas[i])
            for j, idx in enumerate(tri):
                data_tris[i] += data[idx]
                el_avg_tris[i] += self.sketch.trimesh.edge_lengths[idx][tri[(j + 1) % 3]]
        el_avg_tris /= 3
        data_tris /= 3

        tris_picked = np.zeros((trimesh.tris.shape[0]))

        first_seed, bg_mask = self.init_coverage_map_det(tri_areas, eq_edge_tris)
        
        if not (first_seed == -1):
            print("\tUsing iterative masking")
            centroids_labels.append(first_seed)
            centroids[0] = data_tris[first_seed]
            tris_picked[first_seed] = 1
        else:
            print("\tFalling back on corner masking")
            # the first mode of BG masking failed
            # as a fallback, initialize the coverage map with the largest triangles belonging to each corner

            tri_all, tri_tl, tri_tr, tri_bl, tri_br = self.get_largest_corner_tris()
            centroids_labels.append(tri_all)
            centroids[0] = data_tris[tri_all]
            tris_picked[tri_all] = 1
            bg_mask = np.zeros((trimesh.tris.shape[0]))
            first_seed = tri_all

            self.update_coverage(tri_tl, bg_mask, eq_edge_tris)
            self.update_coverage(tri_tr, bg_mask, eq_edge_tris)
            self.update_coverage(tri_bl, bg_mask, eq_edge_tris)
            self.update_coverage(tri_br, bg_mask, eq_edge_tris)

        discovery_maps.append(bg_mask)
        print(f"\tUsing {first_seed} as the first seed. The intial coverage is {np.sum(bg_mask)} out of {trimesh.tris.shape[0]}")
    
        i = 1
        while i < self.n_clusters:

            idx_chosen, min_dists, prob = self.pick_new_centroid_det(data_tris, centroids, centroids_labels, i, area_prob, bg_mask)

            print(f"\tPicked {idx_chosen} with distance weight {min_dists[idx_chosen]} and area weight {area_prob[idx_chosen]} (total weight: {prob[idx_chosen]}) ")
            centroids_labels.append(idx_chosen)
            centroids[i] = data_tris[idx_chosen]
            i += 1
        return centroids, centroids_labels, discovery_maps
    
    def run_kmeans(self, data: np.ndarray, centroids: np.ndarray, centroids_labels: list = None) -> KMeans_res:
        self.paired_v_distances_dict = {}
        labels = np.zeros(data.shape[0], dtype=np.int64)
        centroids_og = centroids.copy()
        unique_ctrd_identifiers = np.arange(centroids.shape[0])
        self._centroid_id_counter = centroids.shape[0]
        for _ in range(self.max_iter):
            # assign points to closest cluster
            labels = self.find_closest_cluster(data, centroids, unique_ctrd_identifiers)

            # update centroids
            centroids_prev = centroids.copy()
            centroids = self.compute_centroids(data, labels)

            # check if centroids didn't change
            same = 0
            for i in range(centroids.shape[0]):
                if np.sum(centroids_prev[i] - centroids[i]) < self.tol:
                    same += 1
                else:
                    unique_ctrd_identifiers[i] = self._centroid_id_counter
                    self._centroid_id_counter += 1
            if same == centroids.shape[0]:
                break
        if _ == self.max_iter - 1 and self.max_iter > 10:
            print("WARNING: Reached max_iter before convergence")
        else:
            print(f"\tThis round finished after {_} iterations with {self._centroid_id_counter - 1} unique centroids ever computed\n")
        return KMeans_res(labels, centroids, centroids_og, centroids_labels)
    
    def assess_error(self, data, centroids, labels):
        total = 0.
        num_classes = np.max(labels) + 1 # assuming labels start from 0
        if not (self.weights is None):
            w = np.sqrt(self.weights)
        for label in range(num_classes):
            sel = labels == label
            if self.weights is None:
                this_error = np.sum(np.power(data[sel] - centroids[label], 2))
            else:
                this_error = np.sum(np.multiply(np.sum(np.power(data[sel] - centroids[label], 2), axis=1), w[sel]))
            total += this_error
        return total

    def fit(self, data: np.ndarray):
        """
            data: a n_samples x n_features matrix
            returns labels for the data
            will also internally set self.centroids
        """
        t1 = time.perf_counter()
        print(f"Daniel's KMeans implementation. Area weights are \n\t{'off' if self.weights is None else 'on'} for centroid updates\n\t{'on' if self.init_method == 'area-weighted' else 'off'} for initialization")
        discovery_maps = None
        if self.init is None:
            res = []
            for _ in range(self.n_init):
                if self.init_method == 'area-weighted':
                    # centroids, centroids_labels, discovery_maps = self.init_centroids(data, self.weights)
                    centroids, centroids_labels, discovery_maps = self.init_centroids_deterministic(data)
                else:
                    centroids, centroids_labels = kmeans_plusplus(data, n_clusters=self.n_clusters)
                res.append(self.run_kmeans(data, centroids, centroids_labels))
            # pick best result
            error = [self.assess_error(data, r.centroids_final, r.labels) for r in res]
            order = np.argsort(error)
            error = [error[idx] for idx in order]
            res = [res[idx] for idx in order]
            best = res[0]
            self.centroids_final = best.centroids_final
            self.labels = best.labels
            res_best = best
        else:
            res = []
            error = []
            res_best = self.run_kmeans(data, self.init)
            self.centroids_final = res_best.centroids_final
            self.labels = res_best.labels
        print(f"Done: took {time.perf_counter() - t1} seconds")
        return (*res_best, res, error, discovery_maps)

    def predict(self, data):
        """
            data: a n_samples x n_features matrix
        """
        
        return self.find_closest_cluster(data, self.centroids_final)
        