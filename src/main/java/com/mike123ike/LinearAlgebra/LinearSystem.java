package com.mike123ike.LinearAlgebra;

public class LinearSystem {
    private final LUDecomposition system;
    private final Matrix originalMatrix;

    public LinearSystem(Matrix matrix) {
        system = new LUDecomposition(matrix);
        originalMatrix = new Matrix(matrix);
    }

    public Matrix getSystem() {
        return new Matrix(originalMatrix);
    }

    public int equations() {
        return system.rows();
    }

    public int variables() {
        return system.cols();
    }

    public VectorSpace solve(Vector V) {
        if (V.size() != system.rows()) {
            throw new IllegalArgumentException("Incompatible vector size for system of equations");
        }
        Matrix L = system.getLReference();
        int[] pivots = system.getPivotReference();
        double[] y = new double[L.rows];
        for (int i = 0; i < L.rows; i++) {
            double sum = 0;
            double[] row = L.getRowReference(i);
            for (int j = 0; j < i; j++) {
                sum += row[j] * y[j];
            }
            y[i] = V.get(pivots[i]) - sum;
        }

        Matrix U = system.getUReference();
        for (int i = 0; i < L.rows; i++) {
            if (i >= U.cols) {
                if (Math.abs(y[i]) >= Matrix.THRESHOLD) {
                    return EmptyVectorSpace.EMPTY_VECTOR_SPACE;
                }
            } else if (Math.abs(U.get(i, i)) < Matrix.THRESHOLD && Math.abs(y[i]) >= Matrix.THRESHOLD) {
                return EmptyVectorSpace.EMPTY_VECTOR_SPACE;
            }
        }

        if (system.getRank() == U.cols) {
            double[] res = new double[U.cols];
            for (int i = U.cols - 1; i >= 0; i--) {
                double sum = 0;
                double[] row = U.getRowReference(i);
                for (int j = i + 1; j < U.cols; j++) {
                    sum += row[j] * res[j];
                }
                res[i] = (y[i] - sum) / row[i];
            }
            return new Vector(res, true);
        }

        int[] pivotCols = new int[U.rows];
        boolean[] isPivot = new boolean[U.cols];
        int rank = 0;

        for (int i = 0; i < U.rows && rank < U.cols; i++) {
            pivotCols[i] = -1;
            double[] row = U.getRowReference(i);
            for (int j = i; j < U.cols; j++) {
                if (Math.abs(row[j]) >= Matrix.THRESHOLD) {
                    pivotCols[i] = j;
                    isPivot[j] = true;
                    rank++;
                    break;
                }
            }
        }

        double[] translation = new double[U.cols];
        for (int i = U.rows - 1; i >= 0; i--) {
            int col = pivotCols[i];
            if (col == -1) {
                continue;
            }
            double sum = 0;
            double[] row = U.getRowReference(i);
            for (int j = col + 1; j < U.cols; j++) {
                sum += row[j] * translation[j];
            }
            translation[col] = (y[i] - sum) / row[col];
        }

        Vector[] basis = new Vector[U.cols - rank];
        int index = 0;
        for (int j = 0; j < U.cols; j++) {
            if (isPivot[j]) {
                continue;
            }
            double[] vector = new double[U.cols];
            vector[j] = 1;
            for (int i = U.rows - 1; i >= 0; i--) {
                int col = pivotCols[i];
                if (col == -1) {
                    continue;
                }
                double sum = 0;
                double[] row = U.getRowReference(i);
                for (int k = col + 1; k < U.cols; k++) {
                    sum += row[k] * vector[k];
                }
                vector[col] = -sum / row[col];
            }
            basis[index++] = new Vector(vector, true);
        }
        return new ParameterizedVector(new Vector(translation, true), basis, true);
    }

    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof LinearSystem)) {
            return false;
        }
        LinearSystem other = (LinearSystem) o;
        return originalMatrix.equals(other.originalMatrix);
    }

    public int hashCode() {
        return originalMatrix.hashCode();
    }

    public String toString() {
        return "Linear System:\n" + originalMatrix.toString();
    }
}
