package data;

import Jama.Matrix;

/**
 *
 * @author Matthieu Nahoum
 */
public class DataSet {

    private Matrix input;
    private Matrix output;

    private DataSet(){

    }

    public static DataSet concatenate(DataSet left, DataSet right){
        DataSet result = null;

        return result;
    }

    public Matrix getModifiedInput(){
        Matrix result = null;

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
