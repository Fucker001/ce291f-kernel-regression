package kernel;

import Jama.*;
import com.gregdennis.drej.GaussianKernel;
import data.*;
import java.util.ArrayList;
import javax.vecmath.GVector;

/**
 *
 * @author Matthieu Nahoum
 */
public class Kernel {

    private DataSet dataset;
    private Matrix input;
    private Matrix output;
    private Matrix VPMatrix;
    private ArrayList<Matrix> VPs = new ArrayList<Matrix>();
    private Matrix K1;
    private final double rho = 0.5;

    public Kernel(DataSet dataset) {
        this.dataset = dataset;
        this.input = this.dataset.getInput();
        this.output = this.dataset.getOutput();

        // compute the kernels
        //this.computeLinearKernel();
        this.computeGaussianKernel(0.2);
        this.computeGaussianKernel(0.7);

        // Build the big VP matrix
        this.computeVPs();
        this.K1 = this.VPMatrix.getMatrix(
                0,
                this.input.getColumnDimension() / 2 - 1,
                0,
                this.VPMatrix.getColumnDimension() - 1);
        System.out.println("VP Matrix: ");
        this.VPMatrix.print(5, 2);
        System.out.println();

        // Print the transpose of the output
        System.out.println("Output transposed: ");
        this.output.transpose().print(5, 2);
        System.out.println();

        QRDecomposition QR = new QRDecomposition(this.VPMatrix);
        System.out.println("R of VPMatrix: ");
        QR.getR().print(5, 2);

        Matrix solution = QR.solve(this.output.transpose());
        System.out.println("Solution: ");
        solution.print(5, 2);

        // Get the optimal U
        //Matrix U = this.K1.solve(this.output);
//        Matrix U = this.VPMatrix.transpose().solve(this.output);
//        U.print(5, 2);
    }

    private void computeVPs() {
        this.VPMatrix = new Matrix(this.input.getColumnDimension(), 0);
        for (Matrix vp : this.VPs) {
            this.VPMatrix = append(this.VPMatrix, vp);
        }
        this.VPMatrix.timesEquals(1.0 / this.rho);
    }

    private Matrix append(Matrix left, Matrix right) throws IllegalArgumentException {
        int dim = left.getRowDimension();
        int numLeft = left.getColumnDimension();
        int numRight = right.getColumnDimension();
        if (right.getRowDimension() != dim) {
            throw new IllegalArgumentException();
        }
        Matrix result = new Matrix(dim, numLeft + numRight);
        // Sets the input and output matrices.
        result.setMatrix(0, dim - 1, 0, numLeft - 1, left);
        result.setMatrix(0, dim - 1, numLeft, numLeft + numRight - 1, right);
        return result;
    }

    private void computeLinearKernel() {
        int size = this.input.getColumnDimension();
        int dim = this.input.getRowDimension();
        Matrix linearKernel = new Matrix(size, size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double value = 0.0;
                for (int k = 0; k < dim; k++) {
                    value += this.input.get(k, i) * this.input.get(k, j);
                }
                linearKernel.set(i, j, value);
            }
        }
        System.out.println("Linear Kernel: ");
        linearKernel.print(5, 2);
        SingularValueDecomposition linearSvd = new SingularValueDecomposition(linearKernel);
        Matrix U = linearSvd.getU();
        System.out.println("Linear Kernel U: ");
        U.print(5, 2);
        Matrix S = linearSvd.getS();
        System.out.println("Linear Kernel S: ");
        S.print(5, 2);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                U.set(i, j, U.get(i, j) * java.lang.Math.sqrt(S.get(j, j)));
            }
        }
        this.VPs.add(U);
        System.out.println("Linear Kernel VPs: ");
        U.print(5, 2);
    }

    private void computeGaussianKernel(double sigma) {
        int size = this.input.getColumnDimension();
        int dim = this.input.getRowDimension();
        Matrix gaussianKernel = new Matrix(size, size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double value = 0.0;
                for (int k = 0; k < dim; k++) {
                    double tmp = this.input.get(k, i) - this.input.get(k, j);
                    value += tmp * tmp;
                }
                value = java.lang.Math.exp(-value / (2.0 * sigma * sigma));
                gaussianKernel.set(i, j, value);
            }
        }
        SingularValueDecomposition linearSvd = new SingularValueDecomposition(gaussianKernel);
        Matrix U = linearSvd.getU();
        System.out.println("Gaussian Kernel, U (sigma = " + sigma + "): ");
        U.print(5, 2);
        Matrix S = linearSvd.getS();
        System.out.println("Gaussian Kernel, S (sigma = " + sigma + "): ");
        S.print(5, 2);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                U.set(i, j, U.get(i, j) * java.lang.Math.sqrt(S.get(j, j)));
            }
        }
        this.VPs.add(U);
        System.out.println("Gaussian Kernel (sigma = " + sigma + "): ");
        U.print(5, 2);
    }

    public Matrix getVPMatrix() {
        return VPMatrix;
    }

    public Matrix getK1() {
        return K1;
    }
}
