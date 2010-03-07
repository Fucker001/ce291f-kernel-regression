/*
 * 
 */
package data;

import Jama.*;
import java.io.File;

/**
 *
 * @author Kev
 */
public class Input {

    private Matrix matrix;

    public static void main(String[] args){
        Input input = new Input();
        System.out.println();
    }

    public Input(){
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

    public Matrix getVector(int index) {
        Matrix result = null;
        result = this.matrix.getMatrix(0, this.matrix.getRowDimension(), index, index);
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
        if (this.matrix.getRowDimension() == other.getDimension()) {
            Matrix concMatrix = new Matrix(this.matrix.getRowDimension(), this.matrix.getColumnDimension() + other.getNumInputs());
            concMatrix.setMatrix(0, this.matrix.getRowDimension() - 1, 0, this.matrix.getColumnDimension() - 1, this.matrix);
            concMatrix.setMatrix(0, this.matrix.getRowDimension() - 1, this.matrix.getColumnDimension(),
                    this.matrix.getColumnDimension() + other.getNumInputs() - 1, this.matrix);
            this.matrix = concMatrix;
        } else {
            throw new IllegalArgumentException();
        }
    }
}

