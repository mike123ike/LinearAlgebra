package com.mike123ike.linalg;

public class LUDecomposition {
    private final Matrix L;
    private final Matrix U;
    private final int[] pivots;
    private final int sign;

    public LUDecomposition(Matrix M) {
        if (M == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }
        int min = Math.min(M.rows, M.cols);
        L = new Matrix(M.rows, min);
        Matrix tempU = new Matrix(M);
        pivots = new int[M.rows];
        for (int i = 0; i < M.rows; i++) {
            pivots[i] = i;
        }
        int sign = 1;

        for (int i = 0; i < min; i++) {
            double[] currentRow = tempU.getRowReference(i);
            int pivot = i;
            double max = Math.abs(currentRow[i]);
            for (int j = i + 1; j < tempU.rows; j++) {
                double val = Math.abs(tempU.get(j, i));
                if (val > max) {
                    max = val;
                    pivot = j;
                }
            }
            if (pivot != i) {
                currentRow = tempU.getRowReference(pivot);
                tempU.swapRows(i, pivot);
                double[] LRow = L.getRowReference(i);
                double[] pivLRow = L.getRowReference(pivot);
                for (int j = 0; j < i; j++) {
                    double tempL = LRow[j];
                    LRow[j] = pivLRow[j];
                    pivLRow[j] = tempL;
                }
                int indexTemp = pivots[i];
                pivots[i] = pivots[pivot];
                pivots[pivot] = indexTemp;
                sign *= -1;
            }
            if (Math.abs(currentRow[i]) < Matrix.THRESHOLD) {
                continue;
            }
            for (int j = i + 1; j < tempU.rows; j++) {
                double[] jRow = tempU.getRowReference(j);
                double scalar = jRow[i] / currentRow[i];
                L.set(j, i, scalar);
                for (int k = i; k < tempU.cols; k++) {
                    jRow[k] -= scalar * currentRow[k];
                }
            }
        }
        this.sign = sign;
        for (int i = 0; i < min; i++) {
            L.set(i, i, 1);
        }
        U = new Matrix(min, M.columns());
        for (int i = 0; i < min; i++) {
            U.setRow(tempU, i);
        }
    }

    // INTERNAL USE ONLY
    Matrix getLReference() {
        return L;
    }

    Matrix getUReference() {
        return U;
    }

    int[] getPivotReference() {
        return pivots;
    }

    public Matrix getL() {
        return new Matrix(L);
    }
    public Matrix getU() {
        return new Matrix(U);
    }
    public int[] getPivots() {
        return pivots.clone();
    }

    public int rows() {
        return L.rows;
    }

    public int cols() {
        return U.cols;
    }

    public int getRank() {
        int sum = 0;
        int min = Math.min(U.rows, U.cols);
        for (int i = 0; i < min; i++) {
            if (Math.abs(U.get(i, i)) >= Matrix.THRESHOLD) {
                sum++;
            }
        }
        return sum;
    }

    public int getSign() {
        return sign;
    }

    public boolean isSquare() {
        return L.rows == U.cols;
    }

    public boolean isSingular() {
        if (!isSquare()) {
            return true;
        }
        for (int i = 0; i < U.cols; i++) {
            if (Math.abs(U.get(i, i)) < Matrix.THRESHOLD) {
                return true;
            }
        }
        return false;
    }

    public double getDeterminant() {
        if (!isSquare()) {
            throw new IllegalStateException("Only square matrices have a determinant");
        }
        double product = sign;
        for (int i = 0; i < U.cols; i++) {
            product *= U.get(i, i);
        }
        return product;
    }

    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof LUDecomposition)) {
            return false;
        }
        LUDecomposition other = (LUDecomposition) o;

        if (sign != other.sign || !L.equals(other.L) || !U.equals(other.U) || pivots.length != other.pivots.length) {
            return false;
        }

        for (int i = 0; i < pivots.length; i++) {
            if (pivots[i] != other.pivots[i]) {
                return false;
            }
        }
        return true;
    }

    public int hashCode() {
        int result = L.hashCode();
        result = 31 * result + U.hashCode();
        for (int i = 0; i < pivots.length; i++) {
            result = 31 * result + Integer.hashCode(pivots[i]);
        }
        result = 31 * result + sign;
        return result;
    }

    @Override
    public String toString() {
        return "LU Decomposition:\n" +
                "L Matrix:\n" + L.toString() +
                "U Matrix:\n" + U.toString() +
                "Pivots: " + java.util.Arrays.toString(pivots) + "\n" +
                "Sign: " + sign;
    }
}