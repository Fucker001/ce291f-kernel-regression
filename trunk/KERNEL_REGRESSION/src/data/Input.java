/*
 * 
 */
package data;

import Jama.*;

/**
 *
 * @author Kev
 */
public class Input {

    private Matrix matrix;

    public static void main(String[] args) {
        Input input = new Input();
        System.out.println();
    }

    /**
     *
     *
     */
    public Input() {
        this.matrix = new Matrix(SampleData.inputs);
    }

    /**
     *
     * @param filePath
     */
    public Input(String filePath) {
    }

    /**
     * 
     * @param url
     * @param user
     * @param password
     */
    public Input(String url, String user, String password) {
    }

    /**
     * @return the matrix
     */
    public Matrix getMatrix() {
        return matrix;
    }

    public double getElement(int i, int j) {
        double result = 0;
        result = this.matrix.get(i, j);
        return result;
    }

    public Matrix getVector(int index) {
        Matrix result = null;
        result = this.matrix.getMatrix(0, this.matrix.getRowDimension() - 1, index, index);
        return result;
    }

    public int getDimension() {
        int result = 0;
        result = this.matrix.getRowDimension();
        return result;
    }

    public int getNumInputs() {
        int result = 0;
        result = this.matrix.getColumnDimension();
        return result;
    }

    public void concatenate(Input other) throws IllegalArgumentException {
        if (this.matrix.getRowDimension() != other.getDimension()) {
            throw new IllegalArgumentException();
        }
        int row = this.matrix.getRowDimension();
        int column = this.matrix.getColumnDimension();
        Matrix concMatrix = new Matrix(row, column + other.getNumInputs());
        concMatrix.setMatrix(0, row - 1, 0, column - 1, this.matrix);
        concMatrix.setMatrix(0, row - 1, column, column + other.getNumInputs() - 1, other.getMatrix());
        this.matrix = concMatrix;
    }

    public static Input concatenate(Input left, Input right) {
        Input result = new Input();
        try {
            result = (Input) left.clone();
        } catch (CloneNotSupportedException ex) {
            ex.printStackTrace();
        }
        result.concatenate(right);
        return result;
    }
}
