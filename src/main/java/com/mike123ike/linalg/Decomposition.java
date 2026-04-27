package com.mike123ike.linalg;

public interface Decomposition {
    int rows();
    int cols();
    double getDeterminant();
    int getRank();
    boolean isSquare();
    boolean isSingular();
}
