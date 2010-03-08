package csv;

import data.DataSet;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.SQLException;

/**
 *
 * @author Matthieu Nahoum
 */
public class DataCSV {

    private DataSet dataset;

    public static void main(String[] args) throws IOException, SQLException {
//        DataCSV d = new DataCSV(new DataSet());
        DataCSV d = new DataCSV(new DataSet("./ttdata.csv","jdbc:postgresql://localhost:5433/kernel", "weather", "weathermm", "2010-03-07", "2010-03-08" ));
    }

    public DataCSV(DataSet dataset) throws IOException {
        this.dataset = dataset;
        this.writeInput();
        this.writeOutput();
    }

    private void writeInput() throws IOException {
        String url = "./input.csv";
        FileWriter f = new FileWriter(url);
        for (int i = 0; i < dataset.getInput().getRowDimension(); i++) {
            String line = "";
            for (int j = 0; j < dataset.getInput().getColumnDimension(); j++) {
                line += dataset.getInput().get(i, j) + ",";
            }
            f.write(line + "\n");
        }
        f.close();
    }

    private void writeOutput() throws IOException {
        String url = "./output.csv";
        FileWriter f = new FileWriter(url);
        String line = "";
        for (int j = 0; j < dataset.getOutput().getColumnDimension(); j++) {
            line += dataset.getOutput().get(0, j) + ",";
        }
        f.write(line + "\n");
        f.close();
    }
}
