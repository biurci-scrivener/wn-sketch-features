#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include "point.h"

struct Data {
    // for one thread
    double result;

    // the following are READ-ONLY
    int * tris;
    double * tri_areas;
    int num_tris;

    struct Point * vertices;
    double * v_fun;
};

void * calc_tv(void * arg) {
    /*
        Calculate TV for one field
    */

    struct Data * data = (struct Data *)arg;

    // for each triangle,
    // calc its gradient
    // then, add sum(abs(grad)) to the total

    double res = 0;

    for (int i = 0; i < data->num_tris; i++) {
        int idx_i = data->tris[3 * i];
        int idx_j = data->tris[3 * i + 1];
        int idx_k = data->tris[3 * i + 2];

        struct Point p_i = data->vertices[idx_i];
        struct Point p_j = data->vertices[idx_j];
        struct Point p_k = data->vertices[idx_k];

        struct Point vec_ki = sub(p_i, p_k);
        struct Point vec_ij = sub(p_j, p_i);

        struct Point di_ki;
        di_ki.x = -vec_ki.y;
        di_ki.y = vec_ki.x;
        struct Point di_ij;
        di_ij.x = -vec_ij.y;
        di_ij.y = vec_ij.x;

        double tri_area = data->tri_areas[i];

        double f_i = data->v_fun[idx_i];
        double f_j = data->v_fun[idx_j];
        double f_k = data->v_fun[idx_k];

        struct Point grad = add( scale(scale(di_ki, 1 / (2 * tri_area)), f_j - f_i) 
                                , scale(scale(di_ij, 1 / (2 * tri_area)), f_k - f_i));
        
        // res += pow(norm(grad), 2) * tri_area;
        res += norm(grad) * tri_area;
    }

    data->result = res;
    return NULL;
}

double * tv(
            int * tris, double * tri_areas, int num_tris,
            double * vertices_d, int num_vertices,
            double * v_funs_d, int num_funs
        ) 
{

    double res = 0 ;
    struct Point * vertices = malloc(sizeof(struct Point) * num_vertices);

    // turn vertices into Points
    for (int i = 0; i < num_vertices; i++) {
        vertices[i].x = vertices_d[i * 2];
        vertices[i].y = vertices_d[(i * 2) + 1];
    }

    // printf("Done building vertices\n");

    double * results = malloc(sizeof(double) * num_funs);

    // parallelize execution
    // by spawning a new thread to deal with field

    pthread_t * threads = malloc(sizeof(pthread_t) * num_funs);
    struct Data * t_data = malloc(sizeof(struct Data) * num_funs);

    for (int i = 0; i < num_funs; i++) {
        // set up data for this field
        t_data[i].v_fun = calloc(num_vertices, sizeof(double));
        for (int j = 0; j < num_vertices; j++) {
            t_data[i].v_fun[j] = v_funs_d[i * num_vertices + j];
        }
        t_data[i].tris = tris;
        t_data[i].tri_areas = tri_areas;
        t_data[i].num_tris = num_tris;

        t_data[i].vertices = vertices;

        // printf("Dispatching thread %i\n", i);
        pthread_create(&threads[i], NULL, calc_tv, (void *)&t_data[i]);
    }

    for (int i = 0; i < num_funs; i++) {
        pthread_join(threads[i], NULL);
        // printf("Thread %i returned\n", i);
        results[i] = t_data[i].result;
        // printf("Saved result of thread %i\n", i);
    }

    // free data
    for (int i = 0; i < num_funs; i++) {
        free(t_data[i].v_fun);
    }
    free(vertices);
    free(threads);
    free(t_data);

    return results;
}