import com.mike123ike.LinearAlgebra.Matrix;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

class MatrixTest {

    private static final double DELTA = 1e-10;

    @Nested
    @DisplayName("1. Constructor and Basic State Tests")
    class BasicTests {
        @Test
        void testDimensions() {
            Matrix m = new Matrix(3, 5);
            assertEquals(3, m.rows());
            assertEquals(5, m.columns());
        }

        @Test
        void testDataConstructorDeepCopy() {
            double[][] data = {{1, 2}, {3, 4}};
            Matrix m = new Matrix(data);
            data[0][0] = 99; // Modify original array
            assertEquals(1.0, m.get(0, 0), DELTA, "Matrix should store a copy, not a reference");
        }

        @Test
        void testCopyConstructorDeepCopy() {
            Matrix original = new Matrix(new double[][]{{1, 1}, {1, 1}});
            Matrix copy = new Matrix(original);
            copy.set(0, 0, 5.0);
            assertNotEquals(original.get(0, 0), copy.get(0, 0), "Modifying copy should not affect original");
        }

        @Test
        void testJaggedArrayException() {
            double[][] jagged = {{1, 2}, {3}};
            assertThrows(IllegalArgumentException.class, () -> new Matrix(jagged));
        }

        @Test
        void testNullConstructorException() {
            assertThrows(IllegalArgumentException.class, () -> new Matrix((double[][]) null));
        }

        @Test
        void testToArray() {
            Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
            double[][] arr = m.toArray();
            arr[0][0] = 0;
            assertEquals(1.0, m.get(0, 0), "toArray should return a deep copy");
        }
    }

    @Nested
    @DisplayName("2. Arithmetic Operations")
    class ArithmeticTests {
        @Test
        void testAdditionSuccess() {
            Matrix a = new Matrix(new double[][]{{1, 2}, {3, 4}});
            Matrix b = new Matrix(new double[][]{{5, 6}, {7, 8}});
            Matrix res = a.add(b);
            assertEquals(6.0, res.get(0, 0), DELTA);
            assertEquals(12.0, res.get(1, 1), DELTA);
        }

        @Test
        void testAdditionDimensionMismatch() {
            Matrix a = new Matrix(2, 2);
            Matrix b = new Matrix(3, 3);
            assertThrows(IllegalArgumentException.class, () -> a.add(b));
        }

        @Test
        void testScalarMultiplication() {
            Matrix m = new Matrix(new double[][]{{1, -2}, {3, 4}});
            Matrix res = m.multiply(-2.0);
            assertEquals(-2.0, res.get(0, 0), DELTA);
            assertEquals(4.0, res.get(0, 1), DELTA);
        }

        @Test
        void testMatrixMultiplicationDotProduct() {
            // 2x3 * 3x2 = 2x2
            Matrix a = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
            Matrix b = new Matrix(new double[][]{{7, 8}, {9, 10}, {11, 12}});
            Matrix res = a.multiply(b);

            // Top Row
            assertEquals(58.0, res.get(0, 0), DELTA);  // (1*7 + 2*9 + 3*11) = 58
            assertEquals(64.0, res.get(0, 1), DELTA);  // (1*8 + 2*10 + 3*12) = 64

            // Bottom Row
            assertEquals(139.0, res.get(1, 0), DELTA); // (4*7 + 5*9 + 6*11) = 139
            assertEquals(154.0, res.get(1, 1), DELTA); // (4*8 + 5*10 + 6*12) = 154
        }

        @Test
        void testTranspose() {
            Matrix m = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
            Matrix t = m.transpose();
            assertEquals(3, t.rows());
            assertEquals(2, t.columns());
            assertEquals(2.0, t.get(1, 0), DELTA);
            assertEquals(4.0, t.get(0, 1), DELTA);
        }
    }

    @Nested
    @DisplayName("3. Linear Algebra Engine (RREF & Rank)")
    class EngineTests {
        @Test
        void testRREFWithRowSwapping() {
            // Needs to swap row 0 and 1 because of the leading 0
            Matrix m = new Matrix(new double[][]{
                    {0, 1, 2},
                    {1, 2, 3}
            });
            m.rrefInPlace();
            assertEquals(1.0, m.get(0, 0), DELTA);
            assertEquals(1.0, m.get(1, 1), DELTA);
            assertEquals(-1.0, m.get(0, 2), DELTA);
        }

        @Test
        void testRREFFullSystem() {
            Matrix m = new Matrix(new double[][]{
                    {1, 2, -1, -4},
                    {2, 3, -1, -11},
                    {-2, 0, -3, 22}
            });
            m.rrefInPlace();
            // Solution: x = -8, y = 1, z = -2
            assertEquals(-8.0, m.get(0, 3), DELTA);
            assertEquals(1.0, m.get(1, 3), DELTA);
            assertEquals(-2.0, m.get(2, 3), DELTA);
        }

        @Test
        void testRankCalculation() {
            Matrix m = new Matrix(new double[][]{
                    {1, 2, 3},
                    {2, 4, 6}, // Dependent
                    {0, 1, 1}
            });
            assertEquals(2, m.getRank());
        }

        @Test
        void testDeterminantAndTrace() {
            Matrix m = new Matrix(new double[][]{{3, 8}, {4, 6}});
            assertEquals(-14.0, m.getDeterminant(), DELTA);
            assertEquals(9.0, m.getTrace(), DELTA);
        }

        @Test
        void testInversionAndSingularCheck() {
            Matrix m = new Matrix(new double[][]{{4, 7}, {2, 6}});
            assertFalse(m.isSingular());
            Matrix inv = m.invert();
            Matrix identity = m.multiply(inv);
            assertEquals(1.0, identity.get(0, 0), DELTA);
            assertEquals(0.0, identity.get(0, 1), DELTA);
        }

        @Test
        void testSingularMatrixException() {
            Matrix singular = new Matrix(new double[][]{{1, 2}, {2, 4}});
            assertTrue(singular.isSingular());
            assertThrows(ArithmeticException.class, singular::invert);
        }
    }

    @Nested
    @DisplayName("4. Binary Exponentiation (pow) Tests")
    class PowerTests {
        @Test
        void testPowerPositive() {
            Matrix m = new Matrix(new double[][]{{1, 1}, {1, 0}}); // Fibonacci
            Matrix res = m.pow(6);
            assertEquals(13.0, res.get(0, 0), DELTA); // F7
        }

        @Test
        void testPowerZeroReturnsIdentity() {
            Matrix m = new Matrix(new double[][]{{2, 3}, {1, 4}});
            Matrix res = m.pow(0);
            assertEquals(1.0, res.get(0, 0), DELTA);
            assertEquals(0.0, res.get(0, 1), DELTA);
        }

        @Test
        void testPowerNegative() {
            Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
            Matrix res = m.pow(-1);
            Matrix expected = m.invert();
            assertEquals(expected.get(0, 0), res.get(0, 0), DELTA);
        }
    }

    @Nested
    @DisplayName("5. Special Matrices & Null Safety")
    class SafetyTests {
        @Test
        void testSingleElementMatrix() {
            Matrix m = new Matrix(new double[][]{{10.0}});
            assertEquals(10.0, m.getDeterminant(), DELTA);
            assertEquals(1, m.getRank());
            assertEquals(0.1, m.invert().get(0, 0), DELTA);
        }

        @Test
        void testAllStaticNullChecks() {
            Matrix m = new Matrix(2, 2);
            assertThrows(IllegalArgumentException.class, () -> Matrix.add(null, m));
            assertThrows(IllegalArgumentException.class, () -> Matrix.multiply(null, 5.0));
            assertThrows(IllegalArgumentException.class, () -> Matrix.transpose(null));
        }

        @Test
        void testThresholdLogic() {
            // 1e-11 is smaller than your 1e-10 threshold
            Matrix m = new Matrix(new double[][]{{1e-11, 0}, {0, 1e-11}});
            assertEquals(0, m.getRank(), "Should treat values below threshold as zero");
            assertTrue(m.isSingular());
        }
    }
}