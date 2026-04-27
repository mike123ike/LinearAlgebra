package com.mike123ike.linalg;

public class QRDecomposition implements Decomposition {
    private final Matrix Q;
    private final Matrix R;
    private final boolean evenReflections;

    Matrix getQReference() {
        return Q;
    }

    Matrix getRReference() {
        return R;
    }

    public QRDecomposition(Matrix M) {
        double[][] qData = new double[M.rows][M.rows];
        double[][] rData = M.toArray();
        boolean reflections = true;
        for (int i = 0; i < M.rows; i++) {
            qData[i][i] = 1;
        }
        double[] v = new double[M.rows];
        int iterations = Math.min(M.rows, M.cols);
        for (int i = 0; i < iterations; i++) {
            double magnitude = 0;
            for (int j = i;  j < M.rows; j++) {
                double val = rData[j][i];
                magnitude += val * val;
            }
            magnitude = Math.sqrt(magnitude) * (rData[i][i] > 0 ? -1 : 1);
            double v0 = rData[i][i] - magnitude;
            double hMagnitude = Math.sqrt(-2 * magnitude * v0);
            if (hMagnitude < Matrix.THRESHOLD) {
                continue;
            }
            reflections = !reflections;
            v[i] = v0 / hMagnitude;
            for (int j = i + 1; j < M.rows; j++) {
                v[j] = rData[j][i] / hMagnitude;
            }
            for (int j = i; j < M.cols; j++) {
                double dot = 0;
                for (int k = i; k < M.rows; k++) {
                    dot += v[k] * rData[k][j];
                }
                dot *= 2;
                for (int k = i; k < M.rows; k++) {
                    rData[k][j] -= v[k] * dot;
                }
            }
            for (int j = 0; j < M.rows; j++) {
                double dot = 0;
                for (int k = i; k < M.rows; k++) {
                    dot += v[k] * qData[k][j];
                }
                dot *= 2;
                for (int k = i; k < M.rows; k++) {
                    qData[k][j] -= v[k] * dot;
                }
            }
        }

        Q = new Matrix(qData, true).transpose();
        R = new Matrix(rData, true);
        evenReflections = reflections;
    }

    public int rows() {
        return R.rows;
    }

    public int cols() {
        return R.cols;
    }

    public Matrix getQ() {
        return new Matrix(Q);
    }

    public Matrix getR() {
        return new Matrix(R);
    }

    public double getDeterminant() {
        if (!isSquare()) {
            throw new IllegalStateException("Only square matrices have a determinant");
        }
        double prod = evenReflections ? 1 : -1;
        for (int i = 0; i < R.rows; i++) {
            prod *= R.get(i, i);
        }
        return prod;
    }

    public int getRank() {
        int rank = 0;
        int min = Math.min(R.rows, R.cols);
        for (int i = 0; i < min; i++) {
            if (Math.abs(R.get(i, i)) >= Matrix.THRESHOLD) {
                rank++;
            }
        }
        return rank;
    }

    public boolean isSquare() {
        return R.isSquare();
    }

    public boolean isSingular() {
        if (!isSquare()) {
            return true;
        }
        for (int i = 0; i < R.rows; i++) {
            if (Math.abs(R.get(i, i)) < Matrix.THRESHOLD) {
                return true;
            }
        }
        return false;
    }

    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof QRDecomposition)) return false;
        QRDecomposition that = (QRDecomposition) o;
        // Parity of reflections must match, and matrices must be equal within THRESHOLD
        return Q.equals(that.Q) && R.equals(that.R);
    }

    public int hashCode() {
        int result = Q.hashCode();
        result = 31 * result + R.hashCode();
        result = 31 * result + (evenReflections ? 1 : 0);
        return result;
    }

    public String toString() {
        return String.format("QR Decomposition:\nQ Matrix:\n%s\nR Matrix:\n%s", R.rows, R.cols, Q.toString(), R.toString());
    }
}
