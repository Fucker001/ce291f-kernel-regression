package data;

import Jama.Matrix;
import database.Database;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Date;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.util.Calendar;
import java.util.GregorianCalendar;

/**
 *
 * @author Kev
 */
public class DataSet {

    private Matrix input;
    private Matrix output;
    private int size;

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

    public DataSet(String dataFilePath, String url, String user, String password, String date_begin, String date_end) throws SQLException, FileNotFoundException, IOException {
        Database db = new Database(url, user, password);
        File f = new File(dataFilePath);

        ResultSet rsSize = db.getResultSet("Select count(*) from weather.weather_data"
                + " WHERE date BETWEEN '" + date_begin + "' AND '" + date_end + "'");
        rsSize.next();
        this.size = rsSize.getInt(1);
        this.input = new Matrix(2, size);
        this.output = new Matrix(1, size);
        // case only dim = 2: hour of the day and humidity
        ResultSet rs = db.getResultSet("SELECT date, humidity FROM weather.weather_data"
                + " WHERE date BETWEEN '" + date_begin + "' AND '" + date_end + "' ORDER BY date ASC");
        java.sql.Timestamp date;
        double humidity;
        int index = 0;
        while (rs.next()) {
            date = rs.getTimestamp(1);
            humidity = new Double(rs.getInt(2));
            long epoch = date.getTime();
            try {
                double tt = this.findTT(dataFilePath, epoch);
                String debug = date.toString();
                double millis = 60 * date.getHours() + date.getMinutes() + date.getSeconds() / 60.0;
                this.input.set(0, index, millis);
                this.input.set(1, index, humidity);
                this.output.set(0, index, tt);
                index++;
            } catch (IOException e) {
                System.out.println(e.getMessage());
            }
        }
        this.input = this.input.getMatrix(0, 1, 0, size - 1);
        this.output = this.output.getMatrix(0, 0, 0, size - 1);
    }

    private double findTT(String dataFilePath, long date) throws FileNotFoundException, IOException {
        double result = -1.0;
        BufferedReader buff = new BufferedReader(new FileReader(dataFilePath));
        String line = buff.readLine();
        long difference;
        while ((line = buff.readLine()) != null) {
            String[] array = line.split(",");
            long ttdate = 1000 * Long.parseLong(array[2].substring(0, array[2].indexOf(".")));
            long tt = Long.parseLong(array[3].substring(0, array[3].indexOf(".")));
            if ((difference = java.lang.Math.abs(ttdate - date)) < 10 * 60 * 1000) {
                return tt;
            }
        }
        size--;
        throw new IOException("Did not find the corresponding travel time.");
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
        result.output.setMatrix(0, 0, 0, colLeft - 1, left.output);
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
