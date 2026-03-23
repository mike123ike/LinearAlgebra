package com.mike123ike.LinearAlgebra;

public class EmptyVectorSpace implements VectorSpace {
    public static final EmptyVectorSpace EMPTY_VECTOR_SPACE = new EmptyVectorSpace();

    private EmptyVectorSpace() {}

    public int getDimension() {
        return -1;
    }

    public Vector getTranslation() {
        throw new UnsupportedOperationException("Empty vector space has no translation");
    }

    public Vector[] getBasis() {
        return new Vector[0];
    }

    public boolean contains(Vector v) {
        return false;
    }

    public boolean equals(Object o) {
        return o instanceof EmptyVectorSpace;
    }

    public int hashCode() {
        return -1;
    }

    public String toString() {
        return "{ Empty Vector Space }";
    }
}
