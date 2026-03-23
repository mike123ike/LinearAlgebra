package com.mike123ike.linalg;

public interface VectorSpace {
    int getDimension();
    Vector getTranslation();
    Vector[] getBasis();
    boolean contains(Vector v);
}
