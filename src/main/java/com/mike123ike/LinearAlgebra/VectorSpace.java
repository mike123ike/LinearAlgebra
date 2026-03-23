package com.mike123ike.LinearAlgebra;

public interface VectorSpace {
    int getDimension();
    Vector getTranslation();
    Vector[] getBasis();
    boolean contains(Vector v);
}
