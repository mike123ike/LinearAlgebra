import com.mike123ike.linalg.*;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class LinearAlgebraTest {

    private static final double DELTA = 1e-10;

    // ===================================================================================
    // 1. EMPTY VECTOR SPACE TESTS
    // ===================================================================================
    @Nested
    class EmptyVectorSpaceTests {
        @Test
        void testSingletonIdentity() {
            EmptyVectorSpace e1 = EmptyVectorSpace.EMPTY_VECTOR_SPACE;
            EmptyVectorSpace e2 = EmptyVectorSpace.EMPTY_VECTOR_SPACE;
            assertSame(e1, e2);
        }

        @Test
        void testProperties() {
            EmptyVectorSpace evs = EmptyVectorSpace.EMPTY_VECTOR_SPACE;
            assertEquals(-1, evs.getDimension());
            assertEquals(0, evs.getBasis().length);
            assertFalse(evs.contains(new Vector(1, 2, 3)));
            assertEquals("{ Empty Vector Space }", evs.toString());
            assertEquals(-1, evs.hashCode());
            assertTrue(evs.equals(EmptyVectorSpace.EMPTY_VECTOR_SPACE));
            assertFalse(evs.equals(new Vector(1)));
        }

        @Test
        void testTranslationThrowsException() {
            EmptyVectorSpace evs = EmptyVectorSpace.EMPTY_VECTOR_SPACE;
            assertThrows(UnsupportedOperationException.class, evs::getTranslation);
        }
    }

    // ===================================================================================
    // 2. VECTOR TESTS
    // ===================================================================================
    @Nested
    class VectorTests {
        @Test
        void testConstructorsAndValidation() {
            assertThrows(IllegalArgumentException.class, () -> new Vector(0));
            assertThrows(IllegalArgumentException.class, () -> new Vector(-5));
            assertThrows(IllegalArgumentException.class, () -> new Vector(new double[0]));

            Vector v1 = new Vector(3);
            assertEquals(3, v1.size());
            assertEquals(0.0, v1.get(2), DELTA);

            Vector v2 = new Vector(1.5, -2.0, 3.14);
            assertEquals(3, v2.size());
            assertEquals(-2.0, v2.get(1), DELTA);

            Vector v3 = new Vector(v2);
            assertEquals(3, v3.size());
            assertEquals(3.14, v3.get(2), DELTA);

            // Ensure deep copy
            v3.set(2, 99.0);
            assertEquals(3.14, v2.get(2), DELTA);
            assertEquals(99.0, v3.get(2), DELTA);
        }

        @Test
        void testToArray() {
            Vector v = new Vector(1, 2, 3);
            double[] arr = v.toArray();
            assertEquals(3, arr.length);
            assertEquals(2.0, arr[1], DELTA);
            arr[1] = 99;
            assertEquals(2.0, v.get(1), DELTA); // Ensure it was a clone
        }

        @Test
        void testVectorSpaceInterface() {
            Vector v = new Vector(4, 5);
            assertEquals(0, v.getDimension());
            assertEquals(0, v.getBasis().length);
            assertEquals(v, v.getTranslation());
            assertNotSame(v, v.getTranslation()); // Should be a copy

            assertTrue(v.contains(new Vector(4, 5)));
            assertFalse(v.contains(new Vector(4, 6)));
            assertFalse(v.contains(new Vector(4, 5, 6)));
        }

        @Test
        void testInPlaceArithmetic() {
            Vector v1 = new Vector(1, 2, 3);
            Vector v2 = new Vector(4, 5, 6);

            v1.addInPlace(v2);
            assertEquals(5.0, v1.get(0), DELTA);
            assertEquals(7.0, v1.get(1), DELTA);
            assertEquals(9.0, v1.get(2), DELTA);

            v1.subtractInPlace(v2);
            assertEquals(1.0, v1.get(0), DELTA);
            assertEquals(2.0, v1.get(1), DELTA);

            v1.multiplyInPlace(2.5);
            assertEquals(2.5, v1.get(0), DELTA);
            assertEquals(5.0, v1.get(1), DELTA);
            assertEquals(7.5, v1.get(2), DELTA);
        }

        @Test
        void testInPlaceArithmeticExceptions() {
            Vector v1 = new Vector(1, 2);
            Vector v2 = new Vector(1, 2, 3);
            assertThrows(IllegalArgumentException.class, () -> v1.addInPlace(v2));
            assertThrows(IllegalArgumentException.class, () -> v1.subtractInPlace(v2));
        }

        @Test
        void testStaticArithmetic() {
            Vector v1 = new Vector(10, 20);
            Vector v2 = new Vector(3, 4);

            Vector sum = Vector.add(v1, v2);
            assertEquals(13.0, sum.get(0), DELTA);
            assertEquals(10.0, v1.get(0), DELTA); // Original unmodified

            Vector diff = Vector.subtract(v1, v2);
            assertEquals(7.0, diff.get(0), DELTA);

            Vector scaled = Vector.multiply(v1, 0.5);
            assertEquals(5.0, scaled.get(0), DELTA);
        }

        @Test
        void testInstanceArithmeticWrappers() {
            Vector v1 = new Vector(1, 2);
            Vector v2 = new Vector(3, 4);
            assertEquals(new Vector(4, 6), v1.add(v2));
            assertEquals(new Vector(-2, -2), v1.subtract(v2));
            assertEquals(new Vector(3, 6), v1.multiply(3));
        }

        @Test
        void testDotProduct() {
            Vector v1 = new Vector(1, 3, -5);
            Vector v2 = new Vector(4, -2, -1);
            assertEquals(3.0, Vector.dotProduct(v1, v2), DELTA);
            assertEquals(3.0, v1.dotProduct(v2), DELTA);

            assertThrows(IllegalArgumentException.class, () -> v1.dotProduct(new Vector(1, 2)));
        }

        @Test
        void testCrossProduct() {
            Vector i = new Vector(1, 0, 0);
            Vector j = new Vector(0, 1, 0);
            Vector k = new Vector(0, 0, 1);

            assertEquals(k, Vector.crossProduct(i, j));
            assertEquals(k, i.crossProduct(j));

            assertEquals(new Vector(0, -1, 0), i.crossProduct(k));
            assertEquals(new Vector(1, 0, 0), j.crossProduct(k));

            Vector v4 = new Vector(1, 2, 3, 4);
            assertThrows(IllegalArgumentException.class, () -> i.crossProduct(v4));
        }

        @Test
        void testMagnitudeAndNormalize() {
            Vector v = new Vector(3, 4);
            assertEquals(5.0, Vector.getMagnitude(v), DELTA);
            assertEquals(5.0, v.getMagnitude(), DELTA);

            Vector norm = Vector.normalize(v);
            assertEquals(0.6, norm.get(0), DELTA);
            assertEquals(0.8, norm.get(1), DELTA);
            assertEquals(1.0, norm.getMagnitude(), DELTA);

            // In place
            v.normalizeInPlace();
            assertEquals(0.6, v.get(0), DELTA);

            // Instance wrapper
            Vector v2 = new Vector(0, 10);
            Vector norm2 = v2.normalize();
            assertEquals(1.0, norm2.get(1), DELTA);
        }

        @Test
        void testEqualsAndHashCode() {
            Vector v1 = new Vector(1, 2, 3);
            Vector v2 = new Vector(1, 2, 3);
            Vector v3 = new Vector(1, 2, 3.0000000000001); // Within threshold
            Vector vDiff = new Vector(1, 2, 4);
            Vector vSize = new Vector(1, 2);

            assertTrue(v1.equals(v1));
            assertTrue(v1.equals(v2));
            assertTrue(v1.equals(v3));
            assertFalse(v1.equals(vDiff));
            assertFalse(v1.equals(vSize));
            assertFalse(v1.equals(null));
            assertFalse(v1.equals("String"));

            assertEquals(v1.hashCode(), v2.hashCode());
        }

        @Test
        void testToString() {
            Vector v = new Vector(1.5, -2.25);
            String str = v.toString();
            assertTrue(str.startsWith("["));
            assertTrue(str.endsWith("]"));
            assertTrue(str.contains("1.5000"));
        }
    }

    // ===================================================================================
    // 3. MATRIX TESTS
    // ===================================================================================
    @Nested
    class MatrixTests {

        @Nested
        class MatrixConstructors {
            @Test
            void testDimensionsConstructor() {
                assertThrows(IllegalArgumentException.class, () -> new Matrix(0, 5));
                assertThrows(IllegalArgumentException.class, () -> new Matrix(5, 0));

                Matrix m = new Matrix(3, 4);
                assertEquals(3, m.rows());
                assertEquals(4, m.columns());
                assertEquals(0.0, m.get(2, 3), DELTA);
            }

            @Test
            void test2DArrayConstructor() {
                assertThrows(IllegalArgumentException.class, () -> new Matrix((double[][]) null));
                assertThrows(IllegalArgumentException.class, () -> new Matrix(new double[0][]));
                assertThrows(IllegalArgumentException.class, () -> new Matrix(new double[][]{{1, 2}, null}));
                assertThrows(IllegalArgumentException.class, () -> new Matrix(new double[][]{{1, 2}, {1}})); // Jagged

                double[][] data = {{1, 2, 3}, {4, 5, 6}};
                Matrix m = new Matrix(data);
                assertEquals(2, m.rows());
                assertEquals(3, m.columns());
                assertEquals(6.0, m.get(1, 2), DELTA);

                data[0][0] = 99; // Ensure deep copy
                assertEquals(1.0, m.get(0, 0), DELTA);
            }

            @Test
            void testCopyConstructor() {
                Matrix original = new Matrix(new double[][]{{1, 2}, {3, 4}});
                Matrix copy = new Matrix(original);
                assertEquals(original, copy);
                original.set(0, 0, 99);
                assertNotEquals(original, copy); // Deep copy verification
            }

            @Test
            void testVectorConstructor() {
                assertThrows(IllegalArgumentException.class, () -> new Matrix((Vector) null, true));

                Vector v = new Vector(1, 2, 3);
                Matrix colMat = new Matrix(v, true);
                assertEquals(3, colMat.rows());
                assertEquals(1, colMat.columns());
                assertEquals(2.0, colMat.get(1, 0), DELTA);

                Matrix rowMat = new Matrix(v, false);
                assertEquals(1, rowMat.rows());
                assertEquals(3, rowMat.columns());
                assertEquals(2.0, rowMat.get(0, 1), DELTA);
            }

            @Test
            void testVectorArrayConstructor() {
                assertThrows(IllegalArgumentException.class, () -> new Matrix((Vector[]) null, true));
                assertThrows(IllegalArgumentException.class, () -> new Matrix(new Vector[0], true));

                Vector v1 = new Vector(1, 2);
                Vector v2 = new Vector(3, 4);
                Vector v3 = new Vector(1, 2, 3);
                assertThrows(IllegalArgumentException.class, () -> new Matrix(new Vector[]{v1, v3}, true));

                Matrix mCols = new Matrix(new Vector[]{v1, v2}, true);
                assertEquals(2, mCols.rows());
                assertEquals(2, mCols.columns());
                assertEquals(3.0, mCols.get(0, 1), DELTA); // First row, second column

                Matrix mRows = new Matrix(new Vector[]{v1, v2}, false);
                assertEquals(2, mRows.rows());
                assertEquals(2, mRows.columns());
                assertEquals(3.0, mRows.get(1, 0), DELTA); // Second row, first column
            }
        }

        @Nested
        class MatrixAccessors {
            @Test
            void testIsSquareAndIsIdentity() {
                Matrix rect = new Matrix(2, 3);
                assertFalse(rect.isSquare());
                assertFalse(rect.isIdentity());

                Matrix sq = new Matrix(2, 2);
                assertTrue(sq.isSquare());
                assertFalse(sq.isIdentity());

                Matrix id = Matrix.identity(3);
                assertTrue(id.isIdentity());

                // Test off-diagonal failure
                Matrix almostId = Matrix.identity(3);
                almostId.set(0, 1, 0.5);
                assertFalse(almostId.isIdentity());

                // Test diagonal failure
                Matrix almostId2 = Matrix.identity(3);
                almostId2.set(1, 1, 0.5);
                assertFalse(almostId2.isIdentity());
            }

            @Test
            void testToArray() {
                Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
                double[][] arr = m.toArray();
                assertEquals(2, arr.length);
                arr[0][0] = 99;
                assertEquals(1.0, m.get(0, 0), DELTA);
            }

            @Test
            void testToVectors() {
                Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
                Vector[] rows = m.toRowVectors();
                assertEquals(new Vector(1, 2), rows[0]);
                assertEquals(new Vector(3, 4), rows[1]);

                Vector[] cols = m.toColumnVectors();
                assertEquals(new Vector(1, 3), cols[0]);
                assertEquals(new Vector(2, 4), cols[1]);
            }
        }

        @Nested
        class Arithmetic {
            @Test
            void testAddAndSubtractInPlace() {
                Matrix m1 = new Matrix(new double[][]{{1, 2}, {3, 4}});
                Matrix m2 = new Matrix(new double[][]{{5, 6}, {7, 8}});
                Matrix m3 = new Matrix(2, 3);

                assertThrows(IllegalArgumentException.class, () -> m1.addInPlace(m3));
                assertThrows(IllegalArgumentException.class, () -> m1.subtractInPlace(m3));

                m1.addInPlace(m2);
                assertEquals(6.0, m1.get(0, 0), DELTA);

                m1.subtractInPlace(m2);
                assertEquals(1.0, m1.get(0, 0), DELTA);
            }

            @Test
            void testStaticArithmetic() {
                Matrix m1 = new Matrix(new double[][]{{1, 2}, {3, 4}});
                Matrix m2 = new Matrix(new double[][]{{5, 6}, {7, 8}});

                assertThrows(IllegalArgumentException.class, () -> Matrix.add(null, m1));

                Matrix sum = Matrix.add(m1, m2);
                assertEquals(6.0, sum.get(0, 0), DELTA);

                Matrix diff = Matrix.subtract(m2, m1);
                assertEquals(4.0, diff.get(0, 0), DELTA);

                Matrix scaled = Matrix.multiply(m1, 2.0);
                assertEquals(2.0, scaled.get(0, 0), DELTA);

                // Instance wrappers
                assertEquals(sum, m1.add(m2));
                assertEquals(diff, m2.subtract(m1));
                assertEquals(scaled, m1.multiply(2.0));
            }

            @Test
            void testMatrixMultiplication() {
                assertThrows(IllegalArgumentException.class, () -> Matrix.multiply(null, new Matrix(1, 1)));
                assertThrows(IllegalArgumentException.class, () -> Matrix.multiply(new Matrix(2, 3), new Matrix(4, 2)));

                Matrix m1 = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
                Matrix m2 = new Matrix(new double[][]{{7, 8}, {9, 10}, {11, 12}});

                Matrix prod = Matrix.multiply(m1, m2);
                assertEquals(2, prod.rows());
                assertEquals(2, prod.columns());
                assertEquals(58.0, prod.get(0, 0), DELTA); // 1*7 + 2*9 + 3*11 = 7+18+33=58
                assertEquals(154.0, prod.get(1, 1), DELTA); // 4*8 + 5*10 + 6*12 = 32+50+72=154

                // Instance wrapper
                assertEquals(prod, m1.multiply(m2));
            }

            @Test
            void testMatrixVectorMultiplication() {
                Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}});
                Vector v = new Vector(7, 8);

                assertThrows(IllegalArgumentException.class, () -> Matrix.multiply(m, new Vector(1, 2, 3)));

                Vector res = Matrix.multiply(m, v);
                assertEquals(3, res.size());
                assertEquals(23.0, res.get(0), DELTA); // 1*7 + 2*8
                assertEquals(53.0, res.get(1), DELTA); // 3*7 + 4*8

                // Instance wrapper
                assertEquals(res, m.multiply(v));
            }
        }

        @Nested
        class AdvancedOperations {
            @Test
            void testTranspose() {
                assertThrows(IllegalArgumentException.class, () -> Matrix.transpose(null));
                Matrix m = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
                Matrix t = m.transpose();
                assertEquals(3, t.rows());
                assertEquals(2, t.columns());
                assertEquals(4.0, t.get(0, 1), DELTA);
            }

            @Test
            void testAugment() {
                Matrix m1 = new Matrix(new double[][]{{1, 2}, {3, 4}});
                Matrix m2 = new Matrix(new double[][]{{5}, {6}});
                Matrix bad = new Matrix(new double[][]{{1}});
                Vector v = new Vector(5, 6);
                Vector badV = new Vector(1);

                assertThrows(IllegalArgumentException.class, () -> Matrix.augment(m1, bad));
                assertThrows(IllegalArgumentException.class, () -> Matrix.augment(m1, badV));

                Matrix aug1 = Matrix.augment(m1, m2);
                assertEquals(3, aug1.columns());
                assertEquals(5.0, aug1.get(0, 2), DELTA);

                Matrix aug2 = Matrix.augment(m1, v);
                assertEquals(3, aug2.columns());
                assertEquals(5.0, aug2.get(0, 2), DELTA);

                assertEquals(aug1, m1.augment(m2));
                assertEquals(aug2, m1.augment(v));
            }

            @Test
            void testTrace() {
                assertThrows(IllegalArgumentException.class, () -> Matrix.getTrace(null));
                assertThrows(IllegalArgumentException.class, () -> Matrix.getTrace(new Matrix(2, 3)));

                Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
                assertEquals(5.0, Matrix.getTrace(m), DELTA);
                assertEquals(5.0, m.getTrace(), DELTA);
            }

            @Test
            void testPow() {
                assertThrows(IllegalArgumentException.class, () -> Matrix.pow(null, 2));
                assertThrows(IllegalArgumentException.class, () -> Matrix.pow(new Matrix(2, 3), 2));

                Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});

                Matrix p0 = Matrix.pow(m, 0);
                assertTrue(p0.isIdentity());

                Matrix p2 = Matrix.pow(m, 2);
                assertEquals(m.multiply(m), p2);

                Matrix pNeg1 = Matrix.pow(m, -1);
                assertEquals(m.invert(), pNeg1);

                Matrix pNeg2 = Matrix.pow(m, -2);
                assertEquals(m.invert().multiply(m.invert()), pNeg2);

                assertEquals(p2, m.pow(2));
            }

            @Test
            void testRREF() {
                assertThrows(IllegalArgumentException.class, () -> Matrix.rref(null));

                // System with unique solution
                Matrix m1 = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
                Matrix rref1 = m1.rref();
                assertEquals(1.0, rref1.get(0, 0), DELTA);
                assertEquals(0.0, rref1.get(0, 1), DELTA);
                assertEquals(-1.0, rref1.get(0, 2), DELTA);
                assertEquals(0.0, rref1.get(1, 0), DELTA);
                assertEquals(1.0, rref1.get(1, 1), DELTA);
                assertEquals(2.0, rref1.get(1, 2), DELTA);

                // Zero matrix
                Matrix zeros = new Matrix(2, 2);
                Matrix rrefZeros = zeros.rref();
                assertEquals(zeros, rrefZeros);

                // Singular matrix
                Matrix singular = new Matrix(new double[][]{{1, 1}, {2, 2}});
                Matrix rrefSingular = singular.rref();
                assertEquals(1.0, rrefSingular.get(0, 0), DELTA);
                assertEquals(1.0, rrefSingular.get(0, 1), DELTA);
                assertEquals(0.0, rrefSingular.get(1, 0), DELTA);
                assertEquals(0.0, rrefSingular.get(1, 1), DELTA);
            }

            @Test
            void testGetRank() {
                assertThrows(IllegalArgumentException.class, () -> Matrix.getRank(null));

                assertEquals(2, Matrix.getRank(new Matrix(new double[][]{{1, 2}, {3, 4}})));
                assertEquals(1, Matrix.getRank(new Matrix(new double[][]{{1, 1}, {2, 2}})));
                assertEquals(0, Matrix.getRank(new Matrix(2, 2))); // Zero matrix
                assertEquals(2, new Matrix(new double[][]{{1, 2}, {3, 4}}).getRank());
            }

            @Test
            void testGetDeterminant() {
                assertThrows(IllegalArgumentException.class, () -> Matrix.getDeterminant(null));
                assertThrows(IllegalArgumentException.class, () -> Matrix.getDeterminant(new Matrix(2, 3)));

                // 1x1
                assertEquals(5.0, Matrix.getDeterminant(new Matrix(new double[][]{{5}})), DELTA);

                // 2x2
                Matrix m2 = new Matrix(new double[][]{{4, 6}, {3, 8}});
                assertEquals(14.0, Matrix.getDeterminant(m2), DELTA);

                // 3x3
                Matrix m3 = new Matrix(new double[][]{{6, 1, 1}, {4, -2, 5}, {2, 8, 7}});
                assertEquals(-306.0, Matrix.getDeterminant(m3), DELTA);

                // Singular det is 0
                Matrix singular = new Matrix(new double[][]{{1, 2}, {2, 4}});
                assertEquals(0.0, Matrix.getDeterminant(singular), DELTA);

                assertEquals(-306.0, m3.getDeterminant(), DELTA);
            }

            @Test
            void testInvert() {
                assertThrows(IllegalArgumentException.class, () -> Matrix.invert(null));
                assertThrows(IllegalArgumentException.class, () -> Matrix.invert(new Matrix(2, 3)));

                Matrix m = new Matrix(new double[][]{{4, 7}, {2, 6}});
                Matrix inv = Matrix.invert(m);

                Matrix prod = m.multiply(inv);
                assertTrue(prod.isIdentity());

                Matrix singular = new Matrix(new double[][]{{1, 2}, {2, 4}});
                assertThrows(ArithmeticException.class, () -> Matrix.invert(singular));

                assertEquals(inv, m.invert());
            }

            @Test
            void testIsSingular() {
                assertThrows(IllegalArgumentException.class, () -> Matrix.isSingular(null));

                assertTrue(Matrix.isSingular(new Matrix(2, 3))); // Non-square are singular
                assertTrue(Matrix.isSingular(new Matrix(new double[][]{{1, 2}, {2, 4}})));
                assertFalse(Matrix.isSingular(new Matrix(new double[][]{{1, 2}, {3, 4}})));

                assertTrue(new Matrix(new double[][]{{1, 2}, {2, 4}}).isSingular());
            }
        }

        @Nested
        class ObjectOverrides {
            @Test
            void testEqualsAndHashCode() {
                Matrix m1 = new Matrix(new double[][]{{1, 2}, {3, 4}});
                Matrix m2 = new Matrix(new double[][]{{1, 2}, {3, 4}});
                Matrix m3 = new Matrix(new double[][]{{1, 2}, {3, 4.00000000000001}}); // Threshold fuzzy check
                Matrix mDiff = new Matrix(new double[][]{{1, 2}, {3, 5}});
                Matrix mSize = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});

                assertTrue(m1.equals(m1));
                assertTrue(m1.equals(m2));
                assertTrue(m1.equals(m3));
                assertFalse(m1.equals(mDiff));
                assertFalse(m1.equals(mSize));
                assertFalse(m1.equals(null));
                assertFalse(m1.equals("String"));

                assertEquals(m1.hashCode(), m2.hashCode());
            }

            @Test
            void testToString() {
                Matrix m = new Matrix(new double[][]{{1.5, -2.5}});
                String str = m.toString();
                assertTrue(str.contains("1.5000"));
                assertTrue(str.contains("-2.5000"));
                assertTrue(str.contains("["));
                assertTrue(str.contains("]"));
            }
        }
    }

    // ===================================================================================
    // 4. LUDECOMPOSITION TESTS
    // ===================================================================================
    @Nested
    class LUDecompositionTests {
        @Test
        void testConstructorNull() {
            assertThrows(IllegalArgumentException.class, () -> new LUDecomposition(null));
        }

        @Test
        void testSquareDecomposition() {
            Matrix m = new Matrix(new double[][]{{4, 3}, {6, 3}});
            LUDecomposition lu = new LUDecomposition(m);

            assertTrue(lu.isSquare());
            assertFalse(lu.isSingular());
            assertEquals(2, lu.getRank());
            assertEquals(2, lu.rows());
            assertEquals(2, lu.cols());
            assertEquals(-6.0, lu.getDeterminant(), DELTA);

            // Check L and U
            Matrix L = lu.getL();
            Matrix U = lu.getU();
            assertEquals(1.0, L.get(0, 0), DELTA);
            assertEquals(0.0, L.get(0, 1), DELTA); // L is lower triangular
            assertEquals(0.0, U.get(1, 0), DELTA); // U is upper triangular
        }

        @Test
        void testWideMatrix() {
            Matrix m = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
            LUDecomposition lu = new LUDecomposition(m);

            assertFalse(lu.isSquare());
            assertTrue(lu.isSingular());
            assertEquals(2, lu.getRank());
            assertThrows(IllegalStateException.class, lu::getDeterminant);

            assertEquals(2, lu.getL().rows());
            assertEquals(2, lu.getL().columns());
            assertEquals(2, lu.getU().rows());
            assertEquals(3, lu.getU().columns());
        }

        @Test
        void testTallMatrix() {
            Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}});
            LUDecomposition lu = new LUDecomposition(m);

            assertFalse(lu.isSquare());
            assertTrue(lu.isSingular());
            assertEquals(2, lu.getRank());
        }

        @Test
        void testSingularSquareMatrix() {
            Matrix m = new Matrix(new double[][]{{1, 2, 3}, {2, 4, 6}, {1, 1, 1}});
            LUDecomposition lu = new LUDecomposition(m);

            assertTrue(lu.isSquare());
            assertTrue(lu.isSingular());
            assertEquals(2, lu.getRank());
            assertEquals(0.0, lu.getDeterminant(), DELTA);
        }

        @Test
        void testPivotsAndSign() {
            Matrix m = new Matrix(new double[][]{{0, 1}, {1, 0}});
            LUDecomposition lu = new LUDecomposition(m);

            assertEquals(-1, lu.getSign()); // One row swap
            int[] pivots = lu.getPivots();
            assertEquals(1, pivots[0]);
            assertEquals(0, pivots[1]);
        }

        @Test
        void testEqualsHashCodeToString() {
            Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
            LUDecomposition lu1 = new LUDecomposition(m);
            LUDecomposition lu2 = new LUDecomposition(m);
            LUDecomposition luDiff = new LUDecomposition(new Matrix(new double[][]{{1, 2}, {3, 5}}));

            assertTrue(lu1.equals(lu1));
            assertTrue(lu1.equals(lu2));
            assertFalse(lu1.equals(luDiff));
            assertFalse(lu1.equals(null));
            assertFalse(lu1.equals("String"));

            assertEquals(lu1.hashCode(), lu2.hashCode());
            assertTrue(lu1.toString().contains("LU Decomposition:"));
        }
    }

    // ===================================================================================
    // 5. LINEAR SYSTEM TESTS
    // ===================================================================================
    @Nested
    class LinearSystemTests {
        @Test
        void testConstructorAndProperties() {
            Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
            LinearSystem sys = new LinearSystem(m);

            assertEquals(2, sys.equations());
            assertEquals(2, sys.variables());
            assertEquals(m, sys.getSystem());
        }

        @Test
        void testSolveValidation() {
            LinearSystem sys = new LinearSystem(new Matrix(2, 2));
            assertThrows(IllegalArgumentException.class, () -> sys.solve(new Vector(1, 2, 3)));
        }

        @Test
        void testSolveUniqueSolution() {
            // 2x + y = 5
            // x - y = 1
            // Sol: x = 2, y = 1
            Matrix m = new Matrix(new double[][]{{2, 1}, {1, -1}});
            LinearSystem sys = new LinearSystem(m);
            VectorSpace sol = sys.solve(new Vector(5, 1));

            assertEquals(0, sol.getDimension());
            assertEquals(new Vector(2, 1), sol.getTranslation());
        }

        @Test
        void testSolveInconsistentSquare() {
            // x + y = 2
            // x + y = 3
            Matrix m = new Matrix(new double[][]{{1, 1}, {1, 1}});
            LinearSystem sys = new LinearSystem(m);
            VectorSpace sol = sys.solve(new Vector(2, 3));

            assertEquals(EmptyVectorSpace.EMPTY_VECTOR_SPACE, sol);
        }

        @Test
        void testSolveInfiniteSolutionsSquare() {
            // x + y = 2
            // 2x + 2y = 4
            Matrix m = new Matrix(new double[][]{{1, 1}, {2, 2}});
            LinearSystem sys = new LinearSystem(m);
            VectorSpace sol = sys.solve(new Vector(2, 4));

            assertTrue(sol instanceof ParameterizedVector);
            assertEquals(1, sol.getDimension());
            assertTrue(sol.contains(new Vector(1, 1)));
            assertTrue(sol.contains(new Vector(2, 0)));
            assertFalse(sol.contains(new Vector(0, 0)));
        }

        @Test
        void testSolveTallConsistent() {
            // x + y = 3
            // x - y = 1
            // 2x + 0y = 4
            // Sol: x=2, y=1
            Matrix m = new Matrix(new double[][]{{1, 1}, {1, -1}, {2, 0}});
            LinearSystem sys = new LinearSystem(m);
            VectorSpace sol = sys.solve(new Vector(3, 1, 4));

            assertEquals(0, sol.getDimension());
            assertEquals(new Vector(2, 1), sol.getTranslation());
        }

        @Test
        void testSolveTallInconsistent() {
            Matrix m = new Matrix(new double[][]{{1, 1}, {1, -1}, {2, 0}});
            LinearSystem sys = new LinearSystem(m);
            VectorSpace sol = sys.solve(new Vector(3, 1, 5)); // Impossible 3rd eq

            assertEquals(EmptyVectorSpace.EMPTY_VECTOR_SPACE, sol);
        }

        @Test
        void testSolveWideConsistent() {
            // x + y + z = 3
            // x - y + z = 1
            Matrix m = new Matrix(new double[][]{{1, 1, 1}, {1, -1, 1}});
            LinearSystem sys = new LinearSystem(m);
            VectorSpace sol = sys.solve(new Vector(3, 1));

            assertTrue(sol instanceof ParameterizedVector);
            assertEquals(1, sol.getDimension()); // 3 variables, rank 2 -> 1 free variable
            assertTrue(sol.contains(new Vector(2, 1, 0))); // A valid particular solution
            assertTrue(sol.contains(new Vector(1, 1, 1))); // Another valid point
        }

        @Test
        void testSolveWideInconsistent() {
            // x + y + z = 1
            // x + y + z = 2
            Matrix m = new Matrix(new double[][]{{1, 1, 1}, {1, 1, 1}});
            LinearSystem sys = new LinearSystem(m);
            VectorSpace sol = sys.solve(new Vector(1, 2));
            assertEquals(EmptyVectorSpace.EMPTY_VECTOR_SPACE, sol);
        }

        @Test
        void testEqualsHashCodeToString() {
            Matrix m1 = new Matrix(new double[][]{{1, 2}, {3, 4}});
            LinearSystem s1 = new LinearSystem(m1);
            LinearSystem s2 = new LinearSystem(m1);
            LinearSystem sDiff = new LinearSystem(new Matrix(new double[][]{{1, 1}, {1, 1}}));

            assertTrue(s1.equals(s1));
            assertTrue(s1.equals(s2));
            assertFalse(s1.equals(sDiff));
            assertFalse(s1.equals(null));
            assertFalse(s1.equals("String"));

            assertEquals(s1.hashCode(), s2.hashCode());
            assertTrue(s1.toString().contains("Linear System:"));
        }
    }

    // ===================================================================================
    // 6. PARAMETERIZED VECTOR TESTS
    // ===================================================================================
    @Nested
    class ParameterizedVectorTests {
        @Test
        void testConstructorsAndProperties() {
            Vector t = new Vector(1, 1);
            Vector[] b = {new Vector(1, 0)};

            ParameterizedVector pv1 = new ParameterizedVector(t, b);
            assertEquals(1, pv1.getDimension());
            assertEquals(t, pv1.getTranslation());
            assertEquals(1, pv1.getBasis().length);
            assertEquals(b[0], pv1.getBasis()[0]);

            // Mutate originals to ensure deep copy
            t.set(0, 99);
            b[0].set(0, 99);
            assertEquals(1.0, pv1.getTranslation().get(0), DELTA);
            assertEquals(1.0, pv1.getBasis()[0].get(0), DELTA);

            ParameterizedVector pvUnique = new ParameterizedVector(new Vector(5, 5));
            assertEquals(0, pvUnique.getDimension());
            assertEquals(new Vector(5, 5), pvUnique.getTranslation());
            assertEquals(0, pvUnique.getBasis().length);
        }

        @Test
        void testContainsLogic() {
            Vector t = new Vector(1, 1, 1);
            Vector b1 = new Vector(1, 0, 0);
            Vector b2 = new Vector(0, 1, 0);
            ParameterizedVector plane = new ParameterizedVector(t, new Vector[]{b1, b2}); // Z=1 plane

            assertThrows(IllegalArgumentException.class, () -> plane.contains(new Vector(1, 1))); // Dimension mismatch

            assertTrue(plane.contains(new Vector(5, -10, 1)));
            assertTrue(plane.contains(new Vector(1, 1, 1)));
            assertFalse(plane.contains(new Vector(5, -10, 2))); // Off plane

            // Unique point contains
            ParameterizedVector point = new ParameterizedVector(new Vector(2, 2));
            assertTrue(point.contains(new Vector(2, 2)));
            assertFalse(point.contains(new Vector(2, 3)));
        }

        @Test
        void testEqualsHashCodeToString() {
            Vector t1 = new Vector(1, 1);
            Vector[] b1 = {new Vector(1, 0)};
            ParameterizedVector pv1 = new ParameterizedVector(t1, b1);
            ParameterizedVector pv2 = new ParameterizedVector(t1, b1);

            Vector tDiff = new Vector(2, 2);
            ParameterizedVector pvDiff1 = new ParameterizedVector(tDiff, b1);

            Vector[] bDiff = {new Vector(0, 1)};
            ParameterizedVector pvDiff2 = new ParameterizedVector(t1, bDiff);

            assertTrue(pv1.equals(pv1));
            assertTrue(pv1.equals(pv2));
            assertFalse(pv1.equals(pvDiff1));
            assertFalse(pv1.equals(pvDiff2));
            assertFalse(pv1.equals(null));
            assertFalse(pv1.equals("String"));

            assertEquals(pv1.hashCode(), pv2.hashCode());
            assertTrue(pv1.toString().contains("Parameterized Vector:"));
            assertTrue(pv1.toString().contains("Translation:"));
            assertTrue(pv1.toString().contains("Basis:"));

            ParameterizedVector pvUnique = new ParameterizedVector(t1);
            assertTrue(pvUnique.toString().contains("Basis: None"));
        }
    }

    @Nested
    class QRDecompositionTests {

        @Test
        void testStandardReconstruction() {
            // A simple 3x2 matrix
            double[][] data = {
                    {12, -51},
                    {6, 167},
                    {-4, 24}
            };
            Matrix A = new Matrix(data);
            QRDecomposition qr = new QRDecomposition(A);

            Matrix Q = qr.getQ();
            Matrix R = qr.getR();

            // Property 1: A = Q * R
            Matrix reconstructed = Q.multiply(R);
            assertTrue(A.equals(reconstructed), "A should equal Q * R");
        }

        @Test
        void testOrthogonalityOfQ() {
            Matrix A = new Matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}});
            QRDecomposition qr = new QRDecomposition(A);
            Matrix Q = qr.getQ();

            // Property 2: Q^T * Q = I
            Matrix QTQ = Q.transpose().multiply(Q);

            // Create Identity Matrix for comparison
            double[][] identity = new double[Q.columns()][Q.columns()];
            for(int i = 0; i < Q.columns(); i++) identity[i][i] = 1.0;
            Matrix I = new Matrix(identity);

            assertTrue(I.equals(QTQ), "Q transpose * Q should be Identity");
        }

        @Test
        void testRIsUpperTriangular() {
            Matrix A = new Matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}});
            QRDecomposition qr = new QRDecomposition(A);
            Matrix R = qr.getR();

            // Property 3: Elements below the diagonal must be 0
            for (int i = 0; i < R.rows(); i++) {
                for (int j = 0; j < R.columns(); j++) {
                    if (i > j) {
                        assertEquals(0.0, R.get(i, j), DELTA,
                                "Element at ["+i+"]["+j+"] should be zero");
                    }
                }
            }
        }

        @Test
        void testRankAndSingularity() {
            // Rank-deficient matrix (col 2 is 2*col 1)
            Matrix rankDeficient = new Matrix(new double[][]{
                    {1, 2},
                    {2, 4},
                    {3, 6}
            });
            QRDecomposition qr = new QRDecomposition(rankDeficient);

            assertEquals(1, qr.getRank(), "Rank should be 1");
            assertFalse(qr.getRank() == 2);
        }

        @Test
        void testDeterminant() {
            // For a square matrix, |A| = |Q| * |R|
            // Since |Q| is 1 or -1 (orthogonal), |A| = ±Product of R's diagonal
            Matrix A = new Matrix(new double[][]{{1, 2}, {3, 4}});
            QRDecomposition qr = new QRDecomposition(A);

            // True determinant of {{1,2},{3,4}} is (1*4 - 2*3) = -2
            assertEquals(-2.0, qr.getDeterminant(), DELTA);
        }

        @Test
        void testThinAndWideMatrices() {
            // "Thin" Matrix (Rows > Cols)
            Matrix thin = new Matrix(new double[5][2]);
            QRDecomposition qrThin = new QRDecomposition(thin);
            assertEquals(5, qrThin.getQ().rows());
            assertEquals(2, qrThin.getR().columns());

            // "Wide" Matrix (Cols > Rows)
            Matrix wide = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
            QRDecomposition qrWide = new QRDecomposition(wide);
            assertTrue(qrWide.getQ().isSquare());
            assertEquals(2, qrWide.getQ().rows());
        }
    }

    @Nested
    class RegressionTests {

        @Test
        void testSimpleLinearRegression() {
            // y = 2x + 1
            Vector x = new Vector(1, 2, 3, 4, 5);
            Vector y = new Vector(3, 5, 7, 9, 11);
            Regression reg = new Regression(x, y);
            Regression.RegressionResult result = reg.polynomialRegression(1);

            // Check prediction
            assertEquals(13.0, result.predict(new Vector(6.0)), DELTA);

            // Check equation coefficients [intercept, slope]
            Vector eq = result.getEquation();
            assertEquals(1.0, eq.get(0), DELTA);
            assertEquals(2.0, eq.get(1), DELTA);
        }

        @Test
        void testMultivariateQuadraticRegression() {
            // z = 1 + x + y + x^2 + xy + y^2
            // Points: (0,0,1), (1,0,3), (0,1,3), (1,1,6)
            Matrix features = new Matrix(new double[][]{
                    {0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 0}
            });
            Vector targets = new Vector(1, 3, 3, 6, 7); // y = 1 + x + y + x^2

            Regression reg = new Regression(features, targets);
            Regression.RegressionResult result = reg.polynomialRegression(2);

            // Test a point not in the training set
            // If x=2, y=2 -> 1 + 2 + 2 + 4 + 4 + 4 = 17 (if all coeffs were 1)
            double prediction = result.predict(new Vector(2, 1));
            assertTrue(prediction > 0);
            assertEquals(result.getDegree(), 2);
        }

        @Test
        void testExponentialRegression() {
            // y = 2 * e^(0.5x) -> linearized as ln(y) = ln(2) + 0.5x
            Vector x = new Vector(0, 1, 2);
            Vector y = new Vector(2.0, 2.0 * Math.exp(0.5), 2.0 * Math.exp(1.0));

            Regression reg = new Regression(x, y);
            Regression.RegressionResult result = reg.exponentialPolynomialRegression(1);

            // The unwarped equation should return the base coefficients
            Vector eq = result.getEquation();
            assertEquals(2.0, eq.get(0), DELTA); // a

            // Predict should return to the original scale
            assertEquals(2.0 * Math.exp(1.5), result.predict(new Vector(3.0)), DELTA);
        }

        @Test
        void testPowerRegressionDomainFailure() {
            Vector x = new Vector(-1, 2, 3); // Negative X
            Vector y = new Vector(1, 4, 9);
            Regression reg = new Regression(x, y);

            Regression.RegressionResult result = reg.powerPolynomialRegression(2);

            // Predicting with a negative number in a log-scale model should return NaN
            assertTrue(Double.isNaN(result.predict(new Vector(-5.0))));
        }

        @Test
        void testLogarithmicRegression() {
            // y = 5 + 2ln(x)
            Vector x = new Vector(1, Math.E, Math.E * Math.E);
            Vector y = new Vector(5, 7, 9);

            Regression reg = new Regression(x, y);
            Regression.RegressionResult result = reg.logarithmicPolynomialRegression(1);

            assertEquals(11.0, result.predict(new Vector(Math.pow(Math.E, 3))), DELTA);
        }

        @Test
        void testResidualsSize() {
            Vector x = new Vector(1, 2, 3);
            Vector y = new Vector(2, 4, 5);
            Regression reg = new Regression(x, y);
            Regression.RegressionResult result = reg.polynomialRegression(1);

            assertEquals(3, result.getResiduals().size());
        }
    }

    @Nested
    class MatrixVectorIntegrationTests {
        @Test
        void testQRToRegressionFlow() {
            // Ensure QR Decomposition handles the Design Matrix correctly
            double[][] data = {{1, 1}, {1, 2}, {1, 3}};
            Matrix m = new Matrix(data);
            QRDecomposition qr = new QRDecomposition(m);

            assertEquals(2, qr.getRank());
            assertTrue(qr.getRank() == 2);
        }
    }

    // ===================================================================================
    // EXTENDED QR DECOMPOSITION TESTS
    // ===================================================================================
    @Nested
    class ExtendedQRDecompositionTests {

        @Test
        void testIdentityMatrix() {
            Matrix I = Matrix.identity(4);
            I.multiplyInPlace(-1);
            QRDecomposition qr = new QRDecomposition(I);

            assertTrue(I.equals(qr.getQ().multiply(qr.getR())), "Q*R of Identity should be Identity");
            assertEquals(1.0, qr.getDeterminant(), DELTA);
            assertEquals(4, qr.getRank());
        }

        @Test
        void testNegativeDeterminant() {
            // Swapping two rows of the identity matrix flips the determinant to -1
            Matrix m = new Matrix(new double[][]{
                    {0, 1, 0},
                    {1, 0, 0},
                    {0, 0, 1}
            });
            QRDecomposition qr = new QRDecomposition(m);

            assertEquals(-1.0, qr.getDeterminant(), DELTA);
        }

        @Test
        void testRankDeficientMatrix() {
            // 3x3 matrix where Row 3 is Row 1 + Row 2
            Matrix m = new Matrix(new double[][]{
                    {1, 2, 3},
                    {4, 5, 6},
                    {5, 7, 9}
            });
            QRDecomposition qr = new QRDecomposition(m);

            assertEquals(2, qr.getRank(), "Rank should be 2 due to linear dependence");
            assertTrue(qr.isSingular());
            assertEquals(0.0, qr.getDeterminant(), DELTA);
        }
    }

    // ===================================================================================
    // EXTENDED REGRESSION TESTS
    // ===================================================================================
    @Nested
    class ExtendedRegressionTests {

        @Test
        void testZeroDegreeRegression() {
            // A degree 0 polynomial regression should return the average of the Y values
            Vector x = new Vector(1, 2, 3, 4, 5);
            Vector y = new Vector(2, 4, 4, 4, 6); // Average is 4.0

            Regression reg = new Regression(x, y);
            Regression.RegressionResult result = reg.polynomialRegression(0);

            Vector eq = result.getEquation();
            assertEquals(1, eq.size());
            assertEquals(4.0, eq.get(0), DELTA);
            assertEquals(4.0, result.predict(new Vector(99.0)), DELTA);
        }

        @Test
        void testUnderdeterminedSystemGracefulDegradation() {
            // We have 2 data points but are asking for a degree 2 fit (requires 3 points: 1, x, x^2)
            // The solver should set the x^2 coefficient to 0 and fit a line.
            Vector x = new Vector(1, 2);
            Vector y = new Vector(2, 4);

            Regression reg = new Regression(x, y);
            Regression.RegressionResult result = reg.polynomialRegression(2);

            Vector eq = result.getEquation();
            assertEquals(3, eq.size()); // [intercept, x, x^2]
            assertEquals(0.0, eq.get(0), DELTA); // intercept = 0
            assertEquals(2.0, eq.get(1), DELTA); // slope = 2
            assertEquals(0.0, eq.get(2), DELTA); // x^2 term is collapsed to 0 due to i >= design.rows check
        }

        @Test
        void testMultivariatePowerRegression() {
            // z = 2 * (x^2) * (y^3)
            // Linearized: ln(z) = ln(2) + 2*ln(x) + 3*ln(y)
            Matrix features = new Matrix(new double[][] {
                    {1, 1},
                    {2, 1},
                    {1, 2},
                    {2, 2},
                    {3, 2}
            });
            Vector targets = new Vector(
                    2 * 1 * 1,
                    2 * 4 * 1,
                    2 * 1 * 8,
                    2 * 4 * 8,
                    2 * 9 * 8
            );

            Regression reg = new Regression(features, targets);
            Regression.RegressionResult result = reg.powerPolynomialRegression(1); // Degree 1 in log-space

            Vector eq = result.getEquation();
            // Equation structure for 2 vars, degree 1: [intercept, x_pow, y_pow]
            assertEquals(2.0, eq.get(0), DELTA); // Intercept is unwarped via Math.exp()
            assertEquals(2.0, eq.get(1), DELTA); // x power remains untouched
            assertEquals(3.0, eq.get(2), DELTA); // y power remains untouched

            double prediction = result.predict(new Vector(3, 3)); // 2 * 9 * 27 = 486
            assertEquals(486.0, prediction, DELTA);
        }
    }
}