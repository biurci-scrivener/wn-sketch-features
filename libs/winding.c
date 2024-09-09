#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include "point.h"

struct Data {
    // for one thread
    double * result;

    // the following are READ-ONLY
    int stroke_id;
    struct Point * stroke_bbs;
    struct Point * stroke_approx;
    
    struct Point * vertices;
    double * angle_min;
    double * angle_max;
    int num_vertices;

    struct Point * vertices_nf;

    int * all_edges;
    int * stroke_ids;
    int num_edges;

    int * parents;

    int * siblings_count;

    int * neighs_parent; // just the first and last one, so has length 2
};

void * calcOneWinding(void * arg) {
    /*
        Calculate the winding number for one query point
    */

    // printf("\tEntered a thread\n");

    struct Data * data = (struct Data *)arg;
    data->result = (double *)calloc(data->num_vertices, sizeof(double));

    // printf("\tI'm here, in the thread for stroke %i\n", data->stroke_id);

    for (int i = 0; i < data->num_vertices; i++) {
        struct Point v1;
        struct Point v2;
        int other;
        int flip;
        int sign;
        double contrib;
        // printf("\t\tCalculating vertex %i\n", i);
        if (bbCheckPt(data->stroke_bbs[data->stroke_id * 2], data->stroke_bbs[data->stroke_id * 2 + 1], data->vertices[i])) { // bbCheck
            for (int edge_idx = 0; edge_idx < data->num_edges; edge_idx++) {
                if (data->stroke_ids[edge_idx] == data->stroke_id) {
                    if (i == 0) {
                        // printf("\tYes, edge %i considered for stroke %i\n", edge_idx, data->stroke_id);
                    }
                    other = -1;
                    int edge_endpt_0 = data->all_edges[edge_idx * 2];
                    int edge_endpt_1 = data->all_edges[edge_idx * 2 + 1];
                    // printf("\t\t\tEdge has endpoints %i and %i\n", edge_endpt_0, edge_endpt_1);

                    if (data->parents[i] == edge_endpt_0) {
                        other = edge_endpt_1;
                        flip = 1;
                    } else if (data->parents[i] == edge_endpt_1) {
                        other = edge_endpt_0;
                        flip = 0;
                    }

                    if (other == -1) {
                        v1 = sub(data->vertices_nf[edge_endpt_0], data->vertices[i]);
                        v2 = sub(data->vertices_nf[edge_endpt_1], data->vertices[i]);
                        normalize(&v1);
                        normalize(&v2);
                        contrib = ( acos(dot(v1, v2)) * (det(v1, v2) > 0 ? 1 : -1) ) / (2 * M_PI);
                        data->result[i] += contrib;
                        // printf("\t\t\tCalc'ed winding number contribution %f for %i against %i\n", contrib, i, edge_idx);
                    } else {
                        // on the stroke boundary
                        if (data->siblings_count[i] > 1) {
                            contrib = (2 * M_PI - angleDiff(data->angle_min[i], data->angle_max[i])) / (4 * M_PI);
                            if (data->neighs_parent[i * 2] == other) {
                                // contrib = 0.25;
                                sign = (flip ? 1 : -1);
                                data->result[i] += contrib * sign;
                            } else if (data->neighs_parent[i * 2 + 1] == other) {
                                // contrib = 0.25;
                                sign = (flip ? -1 : 1);
                                data->result[i] += contrib * sign;
                            }
                        }
                    }
                } 
            }
            // data->result[i] = 0;
        } else {
            // use line segment approximation of stroke
            v1 = sub(data->stroke_approx[data->stroke_id * 2], data->vertices[i]);
            v2 = sub(data->stroke_approx[data->stroke_id * 2 + 1], data->vertices[i]);
            normalize(&v1);
            normalize(&v2);
            contrib = ( acos(dot(v1, v2)) * (det(v1, v2) > 0 ? 1 : -1) ) / (2 * M_PI);
            data->result[i] = contrib;
        }
        
    }

    for (int i = 0; i < data->num_vertices; i++) {
        // printf("\tResult for %i is %f\n", i, data->result[i]);
    }

    return NULL;
}

double * winding(
            double * vertices_d, int num_vertices,
            int * all_edges, int num_edges, 
            int * stroke_ids, double * stroke_bbs_d, 
            double * stroke_approx_d, int num_strokes,
            double * vertices_nf_d, int num_vertices_nf,
            int * parents,
            int * siblings_count,
            int * neighs_parent,
            double * angle_min,
            double * angle_max

        ) 
{

    // printf("\nHello, world!\n");
    // printf("num_vertices: %i\n", num_vertices);
    // printf("num_vertices_nf: %i\n", num_vertices_nf);
    // printf("num_strokes: %i\n", num_strokes);
    // printf("num_edges: %i\n", num_edges);

    struct Point * vertices = malloc(sizeof(struct Point) * num_vertices);
    struct Point * vertices_nf = malloc(sizeof(struct Point) * num_vertices_nf);
    struct Point * stroke_bbs = malloc(sizeof(struct Point) * num_strokes * 2);
    struct Point * stroke_approx = malloc(sizeof(struct Point) * num_strokes * 2);

    // turn vertices and vertices_nf into Points
    for (int i = 0; i < num_vertices; i++) {
        vertices[i].x = vertices_d[i * 2];
        vertices[i].y = vertices_d[(i * 2) + 1];
    }
    for (int i = 0; i < num_vertices_nf; i++) {
        vertices_nf[i].x = vertices_nf_d[i * 2];
        vertices_nf[i].y = vertices_nf_d[(i * 2) + 1];
    }
    for (int i = 0; i < num_strokes; i++) {
        stroke_bbs[i * 2].x = stroke_bbs_d[i * 4];
        stroke_bbs[i * 2].y = stroke_bbs_d[i * 4 + 1];
        stroke_bbs[i * 2 + 1].x = stroke_bbs_d[i * 4 + 2];
        stroke_bbs[i * 2 + 1].y = stroke_bbs_d[i * 4 + 3];

        stroke_approx[i * 2].x = stroke_approx_d[i * 4];
        stroke_approx[i * 2].y = stroke_approx_d[i * 4 + 1];
        stroke_approx[i * 2 + 1].x = stroke_approx_d[i * 4 + 2];
        stroke_approx[i * 2 + 1].y = stroke_approx_d[i * 4 + 3];

        // printf("(%f, %f), (%f, %f)\n", stroke_bbs[i * 2].x, stroke_bbs[i * 2].y, stroke_bbs[i * 2 + 1].x, stroke_bbs[i * 2 + 1].y);
    }

    // printf("Done building vertices and vertices_nf\n");

    double * results = malloc(sizeof(double) * num_strokes * num_vertices);

    // parallelize execution of winding number calc
    // by spawning a new thread to deal with each per-stroke winding number

    pthread_t * threads = malloc(sizeof(pthread_t) * num_strokes);
    struct Data * t_data = malloc(sizeof(struct Data) * num_strokes);

    for (int i = 0; i < num_strokes; i++) {
        // set up data for this stroke
        t_data[i].all_edges = all_edges;
        t_data[i].num_edges = num_edges;
        t_data[i].stroke_ids = stroke_ids;
        t_data[i].stroke_bbs = stroke_bbs;
        t_data[i].stroke_approx = stroke_approx;

        t_data[i].vertices = vertices;
        t_data[i].num_vertices = num_vertices;

        t_data[i].vertices_nf = vertices_nf;
        t_data[i].parents = parents;
        t_data[i].neighs_parent = neighs_parent;
        t_data[i].siblings_count = siblings_count;
        t_data[i].angle_min = angle_min;
        t_data[i].angle_max = angle_max;

        t_data[i].stroke_id = i;

        // printf("Dispatching thread %i\n", i);
        pthread_create(&threads[i], NULL, calcOneWinding, (void *)&t_data[i]);
    }

    for (int i = 0; i < num_strokes; i++) {
        pthread_join(threads[i], NULL);
        // printf("Thread %i returned\n", i);
        for (int j = 0; j < t_data[i].num_vertices; j++) {
            results[i * num_vertices + j] = t_data[i].result[j];
            // printf("Stored %f to results at %i", results[i * num_vertices + j], i * num_vertices + j);
        }
        // printf("Saved result of thread %i\n", i);
    }

    // free data

    free(vertices);
    free(vertices_nf);
    free(threads);
    free(t_data);

    return results;
}