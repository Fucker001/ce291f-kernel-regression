package kernel;

import Jama.*;
import data.*;
import java.util.ArrayList;

/**
 *
 * @author Matthieu Nahoum
 */
public class Kernel {

    private DataSet dataset;
    private Matrix input;
    private Matrix output;
    private Matrix VPMatrix;
    private ArrayList<Matrix> VPs;
    private Matrix K1;
    private final double rho = 0.5;

    public Kernel() {
        this.input = this.dataset.getInput();
        this.output = this.dataset.getOutput();
        this.computeLinearKernel();
        this.computeGaussianKernel(12.6);
        this.computeGaussianKernel(3.2);
        this.computeVPs();
        this.K1 = this.VPMatrix.getMatrix(
                0,
                this.input.getColumnDimension() / 2 - 1,
                0,
                this.VPMatrix.getColumnDimension() - 1);
    }

    private void computeVPs() {
        for (Matrix vp : this.VPs) {
            this.VPMatrix = append(this.VPMatrix, vp);
        }
        this.VPMatrix.times(1.0 / this.rho);
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
        result.setMatrix(0, dim - 1, 0, numRight - 1, left);
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
                    value += this.input.get(i, k) * this.input.get(j, k);
                }
                linearKernel.set(i, j, value);
            }
        }
        SingularValueDecomposition linearSvd = new SingularValueDecomposition(linearKernel);
        Matrix VPs = linearSvd.getU();
        Matrix S = linearSvd.getS();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                VPs.set(i, j, VPs.get(i, j) * java.lang.Math.sqrt(S.get(i, i)));
            }
        }
        this.VPs.add(VPs);
    }

    private void computeGaussianKernel(double sigma) {
        int size = this.input.getColumnDimension();
        int dim = this.input.getRowDimension();
        Matrix gaussianKernel = new Matrix(size, size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double value = 0.0;
                for (int k = 0; k < dim; k++) {
                    double tmp = this.input.get(i, k) - this.input.get(j, k);
                    value += tmp * tmp;
                }
                value = java.lang.Math.exp(-value / (2.0 * sigma * sigma));
                gaussianKernel.set(i, j, value);
            }
        }
        SingularValueDecomposition linearSvd = new SingularValueDecomposition(gaussianKernel);
        Matrix VPs = linearSvd.getU();
        Matrix S = linearSvd.getS();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                VPs.set(i, j, VPs.get(i, j) * java.lang.Math.sqrt(S.get(i, i)));
            }
        }
        this.VPs.add(VPs);
    }

    public Matrix getVPMatrix() {
        return VPMatrix;
    }

    public Matrix getK1() {
        return K1;
    }
}
