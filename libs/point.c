/* functions for manipulating objects defined in point.h */
#include "point.h"
#include <math.h>

#define MAX(a, b) ((a) > (b) ? a : b)
#define MIN(a, b) ((a) < (b) ? a : b)

void normalize(struct Point * p) {
    double mag = sqrt(pow(p->x, 2) + pow(p->y, 2));
    p->x = p->x / mag;
    p->y = p->y / mag;
}

double norm(struct Point p) {
    return sqrt(pow(p.x, 2) + pow(p.y, 2));
}

double dot(struct Point p1, struct Point p2) {
    return MAX(MIN((p1.x * p2.x) + (p1.y * p2.y), 1.0), -1.0);
}

double dot2(struct Point p1, struct Point p2) {
    // unclamped
    return (p1.x * p2.x) + (p1.y * p2.y);
}

double det(struct Point p1, struct Point p2) {
    return (p1.x * p2.y) - (p1.y * p2.x);
}

struct Point add(struct Point p1, struct Point p2) {
    struct Point p;
    p.x = p1.x + p2.x;
    p.y = p1.y + p2.y;
    return p;
}

struct Point sub(struct Point p1, struct Point p2) {
    struct Point p;
    p.x = p1.x - p2.x;
    p.y = p1.y - p2.y;
    return p;
}

struct Point scale(struct Point q, double s) {
    struct Point p;
    p.x = q.x * s;
    p.y = q.y * s;
    return p;
}

double angleDiff(double min_angle, double max_angle) {
    if (min_angle < max_angle) {
        return max_angle - min_angle;
    } else {
        return (2 * M_PI) - (min_angle - max_angle);
    }
}

int bbCheck(struct Point bb1_c1, struct Point bb1_c2, struct Point bb2_c1, struct Point bb2_c2) {
    /*
        c1: xmin, ymin
        c2: xmax, ymax
        not ((o1_xmin > o2_xmax) or (o1_xmax < o2_xmin) or (o1_ymin > o2_ymax) or (o1_ymax < o2_ymin))
    */
    if ( !( (bb1_c1.x > bb2_c2.x) || (bb1_c2.x < bb2_c1.x) || (bb1_c1.y > bb2_c2.y) || (bb1_c2.y < bb2_c1.y) ) ) {
        return 1;
    } else {
        return 0;
    }
}

int bbCheckPt(struct Point bb1_c1, struct Point bb1_c2, struct Point p) {
    /*
        c1: xmin, ymin
        c2: xmax, ymax
        not ((o1_xmin > o2_xmax) or (o1_xmax < o2_xmin) or (o1_ymin > o2_ymax) or (o1_ymax < o2_ymin))
    */
    if ( !( (bb1_c1.x > p.x) || (bb1_c2.x < p.x) || (bb1_c1.y > p.y) || (bb1_c2.y < p.y) ) ) {
        return 1;
    } else {
        return 0;
    }
}