package com.mike123ike.LinearAlgebra;

public class Matrix {
    // Instance variables
    private final double[][] matrix;
    final int rows, cols;

    // Class variable used for rounding
    static final double THRESHOLD = 1e-10;


    // Basic constructor that creates a matrix of all zeros
    public Matrix(int r, int c) {
        if (r <= 0 || c <= 0) {
            throw new IllegalArgumentException("dimensions must be positive");
        }
        rows = r;
        cols = c;
        matrix = new double[rows][cols];
    }

    // Creates matrix using data from a rectangular array
    public Matrix(double[][] data) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("Array cannot be null or empty");
        }
        int firstRowLen = data[0].length;
        for (int i = 1; i < data.length; i++) {
            if (data[i] == null || data[i].length != firstRowLen) {
                throw new IllegalArgumentException("Array is jagged and cannot be turned into a matrix");
            }
        }
        rows = data.length;
        cols = firstRowLen;
        matrix = new double[rows][];
        for (int i = 0; i < rows; i++) {
            matrix[i] = data[i].clone();
        }
    }

    // Creates a copy of another matrix;
    public Matrix(Matrix source) {
        rows = source.rows;
        cols = source.cols;
        matrix = new double[rows][];
        for (int i = 0; i < rows; i++) {
            matrix[i] = source.matrix[i].clone();
        }
    }

    public Matrix(Vector V, boolean columnVector) {
        if (V == null) {
            throw new IllegalArgumentException("Vector cannot be null");
        }
        if (columnVector) {
            rows = V.size();
            cols = 1;
            matrix = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                matrix[i][0] = V.get(i);
            }
        } else {
            rows = 1;
            cols = V.size();
            matrix = new double[rows][];
            matrix[0] = V.toArray();
        }
    }

    public Matrix(Vector[] vectors, boolean asColumns) {
        if (vectors == null || vectors.length == 0) {
            throw new IllegalArgumentException("Vector array cannot be null or empty");
        }
        int expectedLen = vectors[0].size();
        for (int i = 1; i < vectors.length; i++) {
            if (vectors[i].size() != expectedLen) {
                throw new IllegalArgumentException("Vectors have different sizes and cannot be turned into a matrix");
            }
        }
        if (asColumns) {
            rows = expectedLen;
            cols = vectors.length;
            matrix = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    matrix[i][j] = vectors[j].get(i);
                }
            }
        } else {
            rows = vectors.length;
            cols = expectedLen;
            matrix = new double[rows][];
            for (int i = 0; i < rows; i++) {
                matrix[i] = vectors[i].toArray();
            }
        }
    }


    // Methods used for classes in main.java.com.mike123ike.LinearAlgebra to access matrix values

    // DOES NOT CHECK FOR VALID MATRIX
    // ONLY USE TO AVOID MEMORY ALLOCATION FROM ARRAY COPY
    Matrix(double[][] data, boolean usingReference) {
        rows = data.length;
        cols = data[0].length;
        matrix = data;
    }

    void swapRows(int i, int j) {
        double[] temp = matrix[i];
        matrix[i] = matrix[j];
        matrix[j] = temp;
    }

    void setRow(Matrix src, int row ) {
        matrix[row] = src.matrix[row];
    }

    double[] getRowReference(int i) {
        return matrix[i];
    }

    // Accessor methods

    public double get(int row, int col) {
        return matrix[row][col];
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return cols;
    }

    public boolean isSquare() {
        return rows == cols;
    }

    public boolean isIdentity() {
        if (!isSquare()) {
            return false;
        }
        for (int i = 0; i < rows; i++) {
            if (Math.abs(matrix[i][i] - 1) >= THRESHOLD) {
                return false;
            }
            for (int j = 0; j < cols; j++) {
                if (i != j && Math.abs(matrix[i][j]) >= THRESHOLD) {
                    return false;
                }
            }
        }
        return true;
    }

    public double[][] toArray() {
        double[][] res = new double[rows][];
        for (int i = 0; i < rows; i++) {
            res[i] = matrix[i].clone();
        }
        return res;
    }

    public void set(int row, int col, double val) {
        matrix[row][col] = val;
    }

    public Vector[] toRowVectors() {
        Vector[] res = new Vector[rows];
        for (int i = 0; i < rows; i++) {
            res[i] = new Vector(matrix[i]);
        }
        return res;
    }

    public Vector[] toColumnVectors() {
        Vector[] res = new Vector[cols];
        for (int j = 0; j < cols; j++) {
            res[j] = new Vector(rows);
            for (int i = 0; i < rows; i++) {
                res[j].set(i, matrix[i][j]);
            }
        }
        return res;
    }

    // Static factory method that generates NxN identity matrix
    public static Matrix identity(int n) {
        Matrix res = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            res.matrix[i][i] = 1;
        }
        return res;
    }

    public static Matrix diagonal(double[] d) {
        Matrix res = new Matrix(d.length, d.length);
        for (int i = 0; i < res.rows; i++) {
            res.matrix[i][i] = d[i];
        }
        return res;
    }

    // Adds values of other in place
    public void addInPlace(Matrix other) {
        if (rows != other.rows || cols != other.cols) {
            throw new IllegalArgumentException("Expected matrices with same dimensions");
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] += other.matrix[i][j];
            }
        }
    }

    public void subtractInPlace(Matrix other) {
        if (rows != other.rows || cols != other.cols) {
            throw new IllegalArgumentException("Expected matrices with same dimensions");
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] -= other.matrix[i][j];
            }
        }
    }

    // Multiplies all values in the matrix by a scalar
    public void multiplyInPlace(double scalar) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] *= scalar;
            }
        }
    }

    // Transforms matrix into its reduced row-echelon form
    public void rrefInPlace() {
        int lead = 0;
        for (int i = 0; i < rows; i++) {
            int pivot = i;
            while (lead < cols) {
                pivot = i;
                double max = Math.abs(matrix[pivot][lead]);
                for (int j = i + 1; j < rows; j++) {
                    double val = Math.abs(matrix[j][lead]);
                    if (val > max) {
                        max = val;
                        pivot = j;
                    }
                }
                if (max >= THRESHOLD) {
                    break;
                }
                lead++;
            }
            if (lead >= cols) {
                break;
            }
            if (pivot != i) {
                swapRows(i, pivot);
            }
            if (Math.abs(matrix[i][lead] - 1) >= THRESHOLD) {
                double div = matrix[i][lead];
                for (int j = 0; j < cols; j++) {
                    matrix[i][j] /= div;
                }
            }
            for (int row = i + 1; row < rows; row++) {
                if (Math.abs(matrix[row][lead]) < THRESHOLD) {
                    continue;
                }
                double scalar = matrix[row][lead];
                for (int j = 0; j < cols; j++) {
                    matrix[row][j] -= scalar * matrix[i][j];
                }
            }
            lead++;
        }

        for (int i = rows - 1; i >= 0; i--) {
            int pivot = -1;
            for (int j = 0; j < cols; j++) {
                if (Math.abs(matrix[i][j]) >= THRESHOLD) {
                    pivot = j;
                    break;
                }
            }
            if (pivot == -1) {
                continue;
            }
            for (int row = i - 1; row >= 0; row--) {
                double scalar = matrix[row][pivot];
                for (int j = 0; j < cols; j++) {
                    matrix[row][j] -= scalar * matrix[i][j];
                }
            }
        }
    }


    // Returns sum of arguments as a new matrix
    public static Matrix add(Matrix A, Matrix B) {
        if (A == null || B == null) {
            throw new IllegalArgumentException("Matrices cannot be null");
        }
        Matrix res = new Matrix(A);
        res.addInPlace(B);
        return res;
    }

    public static Matrix subtract(Matrix A, Matrix B) {
        Matrix res = new Matrix(A);
        res.subtractInPlace(B);
        return res;
    }

    // Returns scaled matrix as a new object
    public static Matrix multiply(Matrix mat, double scalar) {
        if (mat == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }
        Matrix res = new Matrix(mat);
        res.multiplyInPlace(scalar);
        return res;
    }

    public static Matrix multiply(Matrix A, Matrix B) {
        if (A == null || B == null) {
            throw new IllegalArgumentException("Matrices cannot be null");
        }
        if (A.cols != B.rows) {
            throw new IllegalArgumentException("Incompatible dimensions for matrix multiplication");
        }
        Matrix res = new Matrix(A.rows, B.cols);
        for (int i = 0; i < res.rows; i++) {
            for (int k = 0; k < A.cols; k++) {
                double temp = A.matrix[i][k];
                for (int j = 0; j < res.cols; j++) {
                    res.matrix[i][j] += temp * B.matrix[k][j];
                }
            }
        }
        return res;
    }

    public static Vector multiply(Matrix M, Vector V) {
        if (M.cols != V.size()) {
            throw new IllegalArgumentException("Incompatible dimensions for matrix-vector multiplication");
        }
        Vector res = new Vector(M.rows);
        for (int i = 0; i < M.rows; i++) {
            double rowSum = 0;
            for (int j = 0; j < M.cols; j++) {
                rowSum += V.get(j) * M.matrix[i][j];
            }
            res.set(i, rowSum);
        }
        return res;
    }

    public static Matrix transpose(Matrix mat) {
        if (mat == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }
        Matrix res = new Matrix(mat.cols, mat.rows);
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                res.matrix[j][i] = mat.matrix[i][j];
            }
        }
        return res;
    }

    public static Matrix rref(Matrix mat) {
        if (mat == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }
        Matrix res = new Matrix(mat);
        res.rrefInPlace();
        return res;
    }

    public static double getDeterminant(Matrix mat) {
        if (mat == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }
        if (!mat.isSquare()) {
            throw new IllegalArgumentException("Expected square matrix");
        }
        double product = 1;
        Matrix tempMat = new Matrix(mat);
        for (int i = 0; i < tempMat.rows; i++) {
            int pivot = i;
            double max = Math.abs(tempMat.matrix[pivot][i]);
            for (int j = i + 1; j < tempMat.rows; j++) {
                double val = Math.abs(tempMat.matrix[j][i]);
                if (val > max) {
                    max = val;
                    pivot = j;
                }
            }
            if (max < THRESHOLD) {
                return 0;
            }
            if (pivot != i) {
                tempMat.swapRows(i, pivot);
                product *= -1;
            }
            for (int row = i + 1; row < tempMat.rows; row++) {
                if (Math.abs(tempMat.matrix[row][i]) < THRESHOLD) {
                    continue;
                }
                double scalar = tempMat.matrix[row][i] / tempMat.matrix[i][i];
                for (int j = i; j < tempMat.cols; j++) {
                    tempMat.matrix[row][j] -= scalar * tempMat.matrix[i][j];
                }
            }
            product *= tempMat.matrix[i][i];
        }
        return Math.abs(product) < THRESHOLD ? 0 : product;
    }

    public static Matrix invert(Matrix mat) {
        if (mat == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }
        if (!mat.isSquare()) {
            throw new IllegalArgumentException("Only square matrices can be inverted");
        }
        Matrix augmented = new Matrix(mat.rows, mat.cols * 2);
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                augmented.matrix[i][j] = mat.matrix[i][j];
            }
            augmented.matrix[i][i + mat.cols] = 1;
        }
        augmented.rrefInPlace();
        for (int i = 0; i < mat.rows; i++) {
            if (Math.abs(augmented.matrix[i][i]) < THRESHOLD) {
                throw new ArithmeticException("Matrix is singular and cannot be inverted");
            }
        }
        Matrix res = new Matrix(mat.rows, mat.cols);
        for (int i = 0; i < res.rows; i++) {
            for (int j = 0; j < res.cols; j++) {
                res.matrix[i][j] = augmented.matrix[i][j + mat.cols];
            }
        }
        return res;
    }

    public static Matrix pow(Matrix mat, int n) {
        if (mat == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }
        if (!mat.isSquare()) {
            throw new IllegalArgumentException("Only square matrices can be raised to a power");
        }
        if (n < 0) {
            return pow(mat.invert(), -n);
        }
        Matrix res = identity(mat.rows);
        while (n > 0) {
            if (n % 2 == 1) {
                res = res.multiply(mat);
            }
            mat = mat.multiply(mat);
            n /= 2;
        }
        return res;
    }

    public static int getRank(Matrix mat) {
        if (mat == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }
        Matrix rref = mat.rref();
        int rank = 0;
        for (int i = 0; i < rref.rows; i++) {
            boolean pivot = false;
            for (int j = 0; j < rref.cols; j++) {
                if (Math.abs(rref.matrix[i][j]) >= THRESHOLD) {
                    pivot = true;
                    break;
                }
            }
            if (!pivot) {
                break;
            }
            rank++;
        }
        return rank;
    }

    public static double getTrace(Matrix mat) {
        if (mat == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }
        if (!mat.isSquare()) {
            throw new IllegalArgumentException("Non-square matrices do not have a trace");
        }
        double trace = 0;
        for (int i = 0; i < mat.rows; i++) {
            trace += mat.matrix[i][i];
        }
        return trace;
    }

    public static boolean isSingular(Matrix mat) {
        if (mat == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }
        if (!mat.isSquare()) {
            return true;
        }
        return Math.abs(mat.getDeterminant()) < THRESHOLD;
    }

    public static Matrix augment(Matrix A, Matrix B) {
        if (A.rows != B.rows) {
            throw new IllegalArgumentException("Matrices must have the same number of rows to augment");
        }
        Matrix res = new Matrix(A.rows, A.cols + B.cols);
        for (int i = 0; i < res.rows; i++) {
            for (int j = 0; j < A.cols; j++) {
                res.matrix[i][j] = A.matrix[i][j];
            }
            for (int j = 0; j < B.cols; j++) {
                res.matrix[i][j + A.cols] = B.matrix[i][j];
            }
        }
        return res;
    }

    public static Matrix augment(Matrix M, Vector V) {
        if (M.rows != V.size()) {
            throw new IllegalArgumentException("Vector size must be the same as matrix rows");
        }
        Matrix res = new Matrix(M.rows, M.cols + 1);
        for (int i = 0; i < res.rows; i++) {
            for (int j = 0; j < M.cols; j++) {
                res.matrix[i][j] = M.matrix[i][j];
            }
            res.matrix[i][M.cols] = V.get(i);
        }
        return res;
    }

    public static LUDecomposition decomposeLU(Matrix M) {
        return new LUDecomposition(M);
    }


    // Instance operations
    public Matrix add(Matrix other) {
        return add(this, other);
    }

    public Matrix subtract(Matrix other) {
        return subtract(this, other);
    }

    public Matrix multiply(double scalar) {
        return multiply(this, scalar);
    }

    public Matrix multiply(Matrix other) {
        return multiply(this, other);
    }

    public Vector multiply(Vector other) {
        return multiply(this, other);
    }

    public Matrix transpose() {
        return transpose(this);
    }

    public Matrix rref() {
        return rref(this);
    }

    public double getDeterminant() {
        return getDeterminant(this);
    }

    public Matrix invert() {
        return invert(this);
    }

    public Matrix pow(int n) {
        return pow(this, n);
    }

    public int getRank() {
        return getRank(this);
    }

    public double getTrace() {
        return getTrace(this);
    }

    public boolean isSingular() {
        return isSingular(this);
    }

    public Matrix augment(Matrix other) {
        return augment(this, other);
    }

    public Matrix augment(Vector other) {
        return augment(this, other);
    }

    public LUDecomposition decomposeLU() {
        return decomposeLU(this);
    }

    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Matrix)) {
            return false;
        }
        Matrix other = (Matrix) o;
        if (rows != other.rows || cols != other.cols) {
            return false;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(matrix[i][j] - other.matrix[i][j]) >= THRESHOLD) {
                    return false;
                }
            }
        }
        return true;
    }

    public int hashCode() {
        int result = 17;
        result = 31 * result + rows;
        result = 31 * result + cols;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                long roundedValue = Math.round(matrix[i][j] / THRESHOLD);
                result = 31 * result + Long.hashCode(roundedValue);
            }
        }
        return result;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            sb.append("[ ");
            for (int j = 0; j < cols; j++) {
                sb.append(String.format("%10.4f ", matrix[i][j]));
            }
            sb.append("]\n");
        }
        return sb.toString();
    }
}
