package com.mike123ike.LinearAlgebra;

public class ParameterizedVector implements VectorSpace {
    private final Vector translation;
    private final Vector[] basis;

    ParameterizedVector(Vector t, Vector[] b, boolean useReferences) {
        translation = t;
        basis = b;
    }

    public ParameterizedVector(Vector t, Vector[] b) {
        translation = new Vector(t);
        basis = new Vector[b.length];
        for (int i = 0; i < b.length; i++) {
            basis[i] = new Vector(b[i]);
        }
    }

    public ParameterizedVector(Vector v) {
        translation = new Vector(v);
        basis = new Vector[0];
    }

    public int getDimension() {
        return basis.length;
    }

    public Vector getTranslation() {
        return new Vector(translation);
    }

    public Vector[] getBasis() {
        Vector[] res = new Vector[basis.length];
        for (int i = 0; i < basis.length; i++) {
            res[i] = new Vector(basis[i]);
        }
        return res;
    }

    public boolean contains(Vector v) {
        if (v.size() != translation.size()) {
            throw new IllegalArgumentException("Vector sizes must match");
        }
        if (basis.length == 0) {
            return translation.equals(v);
        }
        Vector w = v.subtract(translation);
        Matrix basisMatrix = new Matrix(basis, true);
        return basisMatrix.getRank() == basisMatrix.augment(w).getRank();
    }

    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ParameterizedVector)) {
            return false;
        }
        ParameterizedVector other = (ParameterizedVector) o;

        if (!translation.equals(other.translation) || basis.length != other.basis.length) {
            return false;
        }

        for (int i = 0; i < basis.length; i++) {
            if (!basis[i].equals(other.basis[i])) {
                return false;
            }
        }
        return true;
    }

    public int hashCode() {
        int result = translation.hashCode();
        for (Vector v : basis) {
            result = 31 * result + v.hashCode();
        }
        return result;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Parameterized Vector:\n");
        sb.append("Translation: ").append(translation.toString()).append("\n");
        if (basis.length > 0) {
            sb.append("Basis:\n");
            for (Vector v : basis) {
                sb.append("  ").append(v.toString()).append("\n");
            }
        } else {
            sb.append("Basis: None (Unique Point)\n");
        }
        return sb.toString();
    }
}
