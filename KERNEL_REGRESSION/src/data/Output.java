package data;

import Jama.*;

/**
 *
 * @author Matthieu Nahoum
 */
public class Output {

    private Matrix matrix;

    public Output() {
        this.matrix = new Matrix(SampleData.outputs);
    }

    /**
     *
     * @param filePath
     */
    public Output(String filePath) {
    }

    /**
     *
     * @param url
     * @param user
     * @param password
     */
    public Output(String url, String user, String password) {
    }
    
}
