package com.mike123ike.linalg;

public class Regression {
    private final Matrix features;
    private final Vector targets;

    public Regression(Matrix f, Vector t) {
        features = new Matrix(f);
        targets = new Vector(t);
    }

    public Regression(Vector f, Vector t) {
        features = new Matrix(f, true);
        targets = new Vector(t);
    }

    public enum Model {
        POLYNOMIAL, EXPONENTIAL, POWER, LOGARITHMIC
    }

    public static class RegressionResult {
        private final Vector equation;
        private final Vector residuals;
        private final Model model;
        private final int degree;
        private final int variables;
        private final String stringForm;
        private final double adjRSquared;

        private RegressionResult(Model t, int v, int d, Vector e, Vector predictions, Vector observed, double rSquared) {
            model = t;
            equation = e;
            residuals = observed.subtract(predictions);
            variables = v;
            degree = d;
            adjRSquared = rSquared;
            StringBuilder sb = new StringBuilder("y = ");
            int[] exponents = new int[variables];
            int[] index = new int[1];
            for (int i = 0; i <= degree; i++) {
                generateString(sb, i, 0, i == 0, exponents, index);
            }
            stringForm = sb.toString();
        }

        private void generateString(StringBuilder eqString, int deg, int var, boolean intercept, int[] exponents, int[] index) {
            if (deg == 0) {
                double val = equation.get(index[0]++);
                if (model == Model.EXPONENTIAL || model == Model.POWER) {
                    val = Math.exp(val);
                }
                appendTerm(eqString, val, exponents, intercept);
                return;
            }
            for (int i = var; i < variables; i++) {
                exponents[i]++;
                generateString(eqString, deg - 1, i, false, exponents, index);
                exponents[i]--;
            }
        }

        private void appendTerm(StringBuilder sb, double val, int[] exponents, boolean intercept) {
            boolean isAdditive = model == Model.POLYNOMIAL || model == Model.LOGARITHMIC;
            boolean firstTerm = sb.length() == 4;
            
            if (isAdditive) {
                if (!firstTerm) {
                    sb.append(val > -Matrix.THRESHOLD ? " + " : " - ");
                } else if (val < -Matrix.THRESHOLD) {
                    sb.append('-');
                }
            } else if (!firstTerm) {
                sb.append(" * ");
            }

            if (model != Model.POWER) {
                val = Math.abs(val);
            }
            String valStr = String.format(val >= 1e-4 || val < Matrix.THRESHOLD ? "%.4f" : "%.2e", val);
            if (intercept) {
                sb.append(valStr);
                return;
            }
            if (model == Model.EXPONENTIAL) {
                sb.append(valStr).append("^(");
            } else if (model == Model.POWER) {
                sb.append('(');
            } else {
                sb.append(valStr);
            }
            boolean first = true;
            for (int i = 0; i < variables; i++) {
                if (exponents[i] == 0) {
                    continue;
                }
                if (!first) {
                    sb.append(' ');
                }
                String varName = variables == 1 ? "x" : "x" + i;
                if (model == Model.LOGARITHMIC || model == Model.POWER) {
                    sb.append("ln(").append(varName).append(')');
                } else {
                    sb.append(varName);
                }
                if (exponents[i] != 1) {
                    sb.append('^').append(exponents[i]);
                }
            }
            if (model == Model.EXPONENTIAL) {
                sb.append(')');
            } else if (model == Model.POWER) {
                sb.append(")^").append(valStr);
            }
        }

        public Vector getEquation() {
            double[] eqVector = equation.toArray();
            if (model == Model.EXPONENTIAL || model == Model.POWER) {
                eqVector[0] = Math.exp(eqVector[0]);
                if (model == Model.EXPONENTIAL) {
                    for (int i = 1; i < eqVector.length; i++) {
                        eqVector[i] = Math.exp(eqVector[i]);
                    }
                }
            }
            return new Vector(eqVector, true);
        }

        public Vector getResiduals() {
            return new Vector(residuals);
        }

        public Model getModel() {
            return model;
        }

        public int getDegree() {
            return degree;
        }

        public double getAdjustedRSquared() {
            return adjRSquared;
        }

        public double predict(Vector inputs) {
            if (inputs.size() != variables) {
                throw new IllegalArgumentException("Unexpected number of variables");
            }
            double[] in = inputs.toArray();
            if (model == Model.LOGARITHMIC || model == Model.POWER) {
                for (int i = 0; i < in.length; i++) {
                    in[i] = Math.log(in[i]);
                }
            }
            double[] design = new double[equation.size()];
            int[] index = new int[1];
            for (int i = 0; i <= degree; i++) {
                generate(design, in, i, 0, 1, index);
            }
            double dotProduct = 0;
            for (int i = 0; i < design.length; i++) {
                dotProduct += design[i] * equation.get(i);
            }
            if (model == Model.EXPONENTIAL || model == Model.POWER) {
                dotProduct = Math.exp(dotProduct);
            }
            return dotProduct;
        }

        public String toString() {
            return stringForm;
        }
    }

    public RegressionResult polynomialRegression(int degree) {
        return regression(features, targets, degree, Model.POLYNOMIAL);
    }

    public RegressionResult logarithmicPolynomialRegression(int degree) {
        int rows = features.rows;
        int cols = features.cols;
        double[][] f = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double[] row = features.getRowReference(i);
            for (int j = 0; j < cols; j++) {
                f[i][j] = Math.log(row[j]);
            }
        }
        return regression(new Matrix(f), targets, degree, Model.LOGARITHMIC);
    }

    public RegressionResult exponentialPolynomialRegression(int degree) {
        double[] v = new double[targets.size()];
        for (int i = 0; i < targets.size(); i++) {
            v[i] = Math.log(targets.get(i));
        }
        return regression(features, new Vector(v), degree, Model.EXPONENTIAL);
    }

    public RegressionResult powerPolynomialRegression(int degree) {
        int rows = features.rows;
        int cols = features.cols;
        double[][] f = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double[] row = features.getRowReference(i);
            for (int j = 0; j < cols; j++) {
                f[i][j] = Math.log(row[j]);
            }
        }
        double[] v = new double[targets.size()];
        for (int i = 0; i < targets.size(); i++) {
            v[i] = Math.log(targets.get(i));
        }
        return regression(new Matrix(f), new Vector(v), degree, Model.POWER);
    }

    private RegressionResult regression(Matrix features, Vector targets, int degree, Model m) {
        int rows = targets.size();
        int vars = features.cols;
        int cols = numCols(vars, degree);
        double[][] data = new double[rows][cols];

        int[] index = new int[1];
        for (int i = 0; i < rows; i++) {
            double[] row = features.getRowReference(i);
            for (int d = 0; d <= degree; d++) {
                generate(data[i], row, d, 0, 1, index);
            }
            index[0] = 0;
        }

        Matrix design = new Matrix(data, true);
        return getRegressionResult(design, targets, m, degree);
    }

    private static void generate(double[] target, double[] vars, int deg, int var, double val, int[] index) {
        if (deg == 0) {
            target[index[0]++] = val;
            return;
        }
        for (int i = var; i < vars.length; i++) {
            generate(target, vars, deg - 1, i, val * vars[i], index);
        }
    }

    private int numCols(int n, int d) {
        long result = 1;
        for (int i = 1; i <= d; i++) {
            result = result * (n + i) / i;
        }
        return (int) result;
    }

    private RegressionResult getRegressionResult(Matrix design, Vector targets, Model model, int deg) {
        QRDecomposition qr = new QRDecomposition(design);
        Vector d = qr.getQReference().transpose().multiply(targets);
        Matrix R = qr.getRReference();
        int n = design.cols;
        double[] equation = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            if (i >= design.rows) {
                continue;
            }
            double sum = 0;
            double[] row = R.getRowReference(i);
            for (int j = i + 1; j < n; j++) {
                sum += row[j] * equation[j];
            }
            equation[i] = Math.abs(row[i]) >= Matrix.THRESHOLD ? (d.get(i) - sum) / row[i] : 0;
        }
        Vector eqVector = new Vector(equation, true);
        Vector p = design.multiply(eqVector);
        if (model == Model.EXPONENTIAL || model == Model.POWER) {
            for (int i = 0; i < p.size(); i++) {
                p.set(i, Math.exp(p.get(i)));
            }
        }

        double sse = 0;
        double sst = 0;
        double mean = 0;
        for (int i = 0; i < targets.size(); i++) {
            mean += targets.get(i);
        }
        mean /= targets.size();
        for (int i = 0; i < targets.size(); i++) {
            double temp = targets.get(i) - p.get(i);
            sse += temp * temp;
            temp = targets.get(i) - mean;
            sst += temp * temp;
        }
        double rSquared = 1 - sse / sst;
        int N = targets.size();

        if (N > design.cols) {
            rSquared = 1 - ((1 - rSquared) * (N - 1)) / (N - design.cols);
        } else {
            rSquared = Double.NaN;
        }
        return new RegressionResult(model, features.cols, deg, eqVector, p, this.targets, rSquared);
    }
}