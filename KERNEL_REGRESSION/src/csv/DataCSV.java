package csv;

import core.DatabaseException;
import core.DatabaseReader;
import core.Time;
import data.DataSet;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;

/**
 *
 * @author Matthieu Nahoum
 */
public class DataCSV {

    private DataSet dataset;
    private DatabaseReader dr;
    private int segmentID;
    private double minTT;
    private String day;
    private ArrayList<Match> data;
    private ArrayList<Weather> weather;
    private Time start;
    private Time end;

    public static void main(String[] args) throws IOException, SQLException {
        //DataCSV d = new DataCSV(new DataSet("./ttdata.csv","jdbc:postgresql://localhost:5433/kernel", "weather", "weathermm", "2010-03-07", "2010-03-08" ));
        DataCSV d = new DataCSV(
                Time.newTimeFromBerkeleyDateTime(2010, 4, 4, 12, 0, 0, 0),
                Time.newTimeFromBerkeleyDateTime(2010, 4, 6, 12, 0, 0, 0),
                3002,
                "",//day
                20.0);
        System.out.println(d.getData().size());
    }

    public DataCSV(Time start, Time end, int segID, String day, double minTT) throws DatabaseException {

        // init
        this.dr = new DatabaseReader();
        this.segmentID = segID;
        this.minTT = minTT;
        this.day = day;
        this.data = new ArrayList<Match>();
        this.weather = new ArrayList<Weather>();
        this.start = start;
        this.end = end;
        // get the weather and filter
        String weather_sql = "SELECT * FROM weather_feed.weather_data, weather_feed.location" + " WHERE weather_data.fk_location_id = location.location_id" + " AND location.location_name = \'Berkeley, CA\'" + " AND date BETWEEN \'" + start.toString() + "\' AND \'" + end.toString() + "\'" + " ORDER BY date ASC";
        this.dr.psCreate("w", weather_sql);
        this.dr.psQuery("w");
        // Loop on the input data and filter
        while (this.dr.psRSNext("w")) {
            // retrieve
            Time time = this.dr.psRSGetTimestamp("w", "date");
            int humidity = this.dr.psRSGetInteger("w", "humidity");
            int temperature = this.dr.psRSGetInteger("w", "temperature");
            String status = this.dr.psRSGetVarChar("w", "status");
            //store
            this.weather.add(new Weather(time, humidity, status, temperature));
        }
        // get the data and filter
        String sql = "SELECT * FROM sensys_feed.match" + " WHERE date BETWEEN \'" + start.toString() + "\' AND \'" + end.toString() + "\'" + " AND fk_segment_id = " + this.segmentID + " ORDER BY date ASC";
        this.dr.psCreate("q", sql);
        this.dr.psQuery("q");
        // Loop on the input data and filter
        while (this.dr.psRSNext("q")) {
            //retrieve
            Time time = this.dr.psRSGetTimestamp("q", "date");
            double tt = this.dr.psRSGetDouble("q", "travel_time");
            // filter
            if (tt >= 0 && (time.getDisplayNameDayLong().equals(this.day) || this.day.equals(""))) {
                // find corresponding weather
                int count = 0;
                Weather myWeather = new Weather(Time.newTimeFromBerkeleyDateTime(2900, 1, 1, 0, 0, 0, 0), 0, "", 0);
                for (Weather wth : this.weather) {
                    count++;
                    Time wTime = wth.getDate();
                    if (wTime.after(time) && myWeather.getDate().after(wTime)) {
                        myWeather = wth;
                    }
                }
                // store
                this.data.add(new Match(time, tt, myWeather));
            }
        }
        try {
            this.write();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private String getDescription() {
        String result = "";
        String sql = "SELECT description FROM sensys_feed.segment_prop"
                + " WHERE segment_id = " + this.segmentID;
        try {
            this.dr.psCreate("d", sql);
            this.dr.psQuery("d");
            while (this.dr.psRSNext("d")) {
                result = this.dr.psRSGetVarChar("d", "description");
            }
        } catch (DatabaseException ex) {
            ex.printStackTrace();
        }
        return result;
    }

    private void write() throws IOException {
        String urlI = "./input.csv";
        String urlO = "./output.csv";
        String urlDescription = "./description.txt";

        FileWriter fI = new FileWriter(urlI);
        FileWriter fO = new FileWriter(urlO);
        FileWriter fDescription = new FileWriter(urlDescription);

        String lineI1 = "";
        String lineI2 = "";
        String lineI3 = "";
        String lineO = "";
        String lineDescription = this.getDescription() + ","
                + " start=" + this.start.toString() + ", end=" + this.end.toString();

        for (Match match : this.data) {
            lineI1 += match.getDateDouble() + ","; // time in the day
            lineI2 += match.getWeather().getTemperature() + ","; // temperature
            lineI3 += match.getWeather().getStatusDouble() + ","; // status
            lineO += match.getTravelTime() + ","; // travel time, output
        }
        lineI1 += "\n";
        lineI2 += "\n";

        fI.write(lineI1);
        fI.write(lineI2);
        fI.write(lineI3);
        fO.write(lineO);
        fDescription.write(lineDescription);
        fI.close();
        fO.close();
        fDescription.close();
    }

    public ArrayList<Match> getData() {
        return data;
    }
}
