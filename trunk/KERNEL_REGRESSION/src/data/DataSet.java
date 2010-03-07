package data;

import Jama.Matrix;

/**
 *
 * @author Matthieu Nahoum
 */
public class DataSet {

    private Matrix input;
    private Matrix output;

    public static void main(String[] args) {
        DataSet dataset1 = new DataSet();
        DataSet dataset2 = new DataSet();
        DataSet dataset3 = concatenate(dataset1, dataset2);
        System.out.println();
    }

    public DataSet() {
        this.input = new Matrix(SampleData.inputs);
        this.output = new Matrix(SampleData.outputs);
    }

    /**
     * @return the concatenated data set.
     */
    public static DataSet concatenate(DataSet left, DataSet right) throws IllegalArgumentException {
        DataSet result = new DataSet();
        if (left.input.getRowDimension() != right.input.getRowDimension()) {
            throw new IllegalArgumentException();
        }
        // Gets the dimensions of input and output matrix.
        int dim = left.input.getRowDimension();
        int colLeft = left.input.getColumnDimension();
        int colRight = right.input.getColumnDimension();

        // Initializes the concatenated input and output matrices.
        result.input = new Matrix(dim, colLeft + colRight);
        result.output = new Matrix(1, colLeft + colRight);

        // Sets the input and output matrices.
        result.input.setMatrix(0, dim - 1, 0, colLeft - 1, left.input);
        result.input.setMatrix(0, dim - 1, colLeft, colLeft + colRight - 1, right.input);
        result.output.setMatrix(0, 0, 0, colRight - 1, left.output);
        result.output.setMatrix(0, 0, colLeft, colLeft + colRight - 1, right.output);

        return result;
    }

    public Matrix getModifiedInput() {
        Matrix result = null;
        Matrix ones = null;
        result = new Matrix(input.getRowDimension() + 1, input.getColumnDimension());
        ones = new Matrix(1, input.getColumnDimension(), 1);

        result.setMatrix(0, input.getRowDimension() - 1, 0, input.getColumnDimension() - 1, input);
        result.setMatrix(input.getRowDimension(), input.getRowDimension(), 0, input.getColumnDimension() - 1, ones);

        return result;
    }

    /**
     * @return the input
     */
    public Matrix getInput() {
        return input;
    }

    /**
     * @return the output
     */
    public Matrix getOutput() {
        return output;
    }
}
