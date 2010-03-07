/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package data;

import Jama.*;

/**
 *
 * @author Kev
 */
public class Input {

    private Matrix matrix;

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

    private void load() {
    }

    public Matrix getVector(int index) {
        Matrix result = null;
        result = matrix.getMatrix(0, matrix.getRowDimension(), index, index);
        return result;
    }

    public int getDimension() {
        int result = 0;
        result = matrix.getRowDimension();
        return result;
    }

    public int getNumInputs() {
        int result = 0;
        result = matrix.getColumnDimension();
        return result;
    }

    public void concatenate(Input other) throws IllegalArgumentException {
        throw new IllegalArgumentException();
    }
}

