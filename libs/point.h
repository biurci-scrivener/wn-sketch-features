/* structs for manipulating sketch objects */

struct Point {
    double x;
    double y;
};

struct Edge {
    // will also reuse same struct for Ray,
    // where p1 is origin and p2 is direction
    struct Point p1;
    struct Point p2;
};

struct Intersection {
    int sign;
    struct Point p;
    double t; 
};

void normalize(struct Point * p);
double norm(struct Point p);
double dot(struct Point p1, struct Point p2);
double dot2(struct Point p1, struct Point p2);
double det(struct Point p1, struct Point p2);
struct Point add(struct Point p1, struct Point p2);
struct Point sub(struct Point p1, struct Point p2);
struct Point scale(struct Point q, double s);
double angleDiff(double angle_min, double angle_max);
int bbCheck(struct Point bb1_c1, struct Point bb1_c2, struct Point bb2_c1, struct Point bb2_c2);
int bbCheckPt(struct Point bb1_c1, struct Point bb1_c2, struct Point p);