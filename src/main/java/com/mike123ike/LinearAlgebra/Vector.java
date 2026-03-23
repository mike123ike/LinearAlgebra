package com.mike123ike.LinearAlgebra;

public class Vector implements VectorSpace {
    private final double[] vector;
    private final int size;

    public Vector(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("Size must be positive");
        }
        vector = new double[n];
        size = n;
    }

    public Vector(double... v) {
        if (v.length == 0) {
            throw new IllegalArgumentException("Vector size must be positive");
        }
        size = v.length;
        vector = v.clone();
    }

    public Vector(Vector v) {
        size = v.size;
        vector = v.vector.clone();
    }

    // INTERNAL USE constructor to avoid cloning array
    Vector(double[] v, boolean usingReference) {
        size = v.length;
        vector = v;
    }

    public int size() {
        return size;
    }

    public double get(int i) {
        return vector[i];
    }

    public double[] toArray() {
        return vector.clone();
    }

    public void set(int i, double value) {
        vector[i] = value;
    }

    public int getDimension() {
        return 0;
    }

    public Vector getTranslation() {
        return new Vector(this);
    }

    public Vector[] getBasis() {
        return new Vector[0];
    }

    public boolean contains(Vector v) {
        return this.equals(v);
    }

    public void addInPlace(Vector other) {
        if (size != other.size) {
            throw new IllegalArgumentException("Vectors must have same size to add");
        }
        for (int i = 0; i < size; i++) {
            vector[i] += other.vector[i];
        }
    }

    public void subtractInPlace(Vector other) {
        if (size != other.size) {
            throw new IllegalArgumentException("Vectors must have the same size to subtract");
        }
        for (int i = 0; i < size; i++) {
            vector[i] -= other.vector[i];
        }
    }

    public void multiplyInPlace(double scalar) {
        for (int i = 0; i < size; i++) {
            vector[i] *= scalar;
        }
    }

    public void normalizeInPlace() {
        multiplyInPlace(1 / getMagnitude());
    }

    public static Vector add(Vector A, Vector B) {
        Vector res = new Vector(A);
        res.addInPlace(B);
        return res;
    }

    public static Vector subtract(Vector A, Vector B) {
        Vector res = new Vector(A);
        res.subtractInPlace(B);
        return res;
    }

    public static Vector multiply(Vector v, double scalar) {
        Vector res = new Vector(v);
        res.multiplyInPlace(scalar);
        return res;
    }

    public static double dotProduct(Vector A, Vector B) {
        if (A.size != B.size) {
            throw new IllegalArgumentException("Vectors must have the same size to take dot product");
        }
        double sum = 0;
        for (int i = 0; i < A.size; i++) {
            sum += A.vector[i] * B.vector[i];
        }
        return sum;
    }

    public static Vector crossProduct(Vector A, Vector B) {
        if (A.size != 3 || B.size != 3) {
            throw new IllegalArgumentException("Vectors must have a size of 3 to take cross product");
        }
        double i = A.vector[1] * B.vector[2] - A.vector[2] * B.vector[1];
        double j = A.vector[2] * B.vector[0] - A.vector[0] * B.vector[2];
        double k = A.vector[0] * B.vector[1] - A.vector[1] * B.vector[0];
        return new Vector(i, j, k);
    }

    public static double getMagnitude(Vector v) {
        return Math.sqrt(dotProduct(v, v));
    }

    public static Vector normalize(Vector v) {
        Vector res = new Vector(v);
        res.normalizeInPlace();
        return res;
    }


    public Vector add(Vector other) {
        return add(this, other);
    }

    public Vector subtract(Vector other) {
        return subtract(this, other);
    }

    public Vector multiply(double scalar) {
        return multiply(this, scalar);
    }

    public double dotProduct(Vector other) {
        return dotProduct(this, other);
    }

    public Vector crossProduct(Vector other) {
        return crossProduct(this, other);
    }

    public double getMagnitude() {
        return getMagnitude(this);
    }

    public Vector normalize() {
        return normalize(this);
    }

    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Vector)) {
            return false;
        }
        Vector other = (Vector) o;
        if (size != other.size) {
            return false;
        }
        for (int i = 0; i < size; i++) {
            if (Math.abs(vector[i] - other.vector[i]) >= Matrix.THRESHOLD) {
                return false;
            }
        }
        return true;
    }

    public int hashCode() {
        int result = 17;
        result = 31 * result + size;
        for (int i = 0; i < size; i++) {
            // Divide by threshold and round to group nearby floating point values together
            long roundedValue = Math.round(vector[i] / Matrix.THRESHOLD);
            result = 31 * result + Long.hashCode(roundedValue);
        }
        return result;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[ ");
        for (int i = 0; i < size; i++) {
            sb.append(String.format("%10.4f ", vector[i]));
        }
        sb.append("]");
        return sb.toString();
    }
}
