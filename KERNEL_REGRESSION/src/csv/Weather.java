package csv;

import core.Time;

/**
 *
 * @author Matthieu Nahoum
 */
public class Weather {

    private Time date;
    private double humidity;
    private String status;
    private double statusDouble;
    private double temperature;

    public static void main(String[] args) {
        Time t1 = Time.newTimeFromBerkeleyDateTime(2010, 4, 2, 0, 0, 0, 0);
        Time t2 = Time.newTimeFromBerkeleyDateTime(2010, 4, 8, 23, 59, 0, 0);
        boolean t1After = t1.after(t2);
        boolean t2After = t2.after(t1);
        System.out.println(t1After);
        System.out.println(t2After);
    }

    public Weather(Time date, int humidity, String status, int temperature) {
        this.date = date;
        this.humidity = humidity / 100.0;
        this.temperature = temperature / 100.0;
        this.status = status;
        this.statusDouble = this.getTheStatus(status);
    }

    public Weather(Time date, double humidity, String status, double temperature) {
        this.date = date;
        this.humidity = humidity;
        this.temperature = temperature;
        this.status = status;
        this.statusDouble = this.getTheStatus(status);
    }

    private double getTheStatus(String status) {
        if (status.equals("Sunny")) {
            return 0.1;
        }
        if (status.equals("Clear")) {
            return 0.12;
        }
        if (status.equals("Partly Cloudy")) {
            return 0.2;
        }
        if (status.equals("Cloudy")) {
            return 0.25;
        }
        if (status.equals("Mostly Cloudy")) {
            return 0.3;
        }
        if (status.equals("Overcast")) {
            return 0.35;
        }
        if (status.equals("Haze")) {
            return 0.4;
        }
        if (status.equals("Light Rain")) {
            return 0.5;
        }
        if (status.equals("Showers")) {
            return 0.6;
        }
        if (status.equals("Rain")) {
            return 0.8;
        }
        return 0.0;
    }

    public Time getDate() {
        return date;
    }

    public double getHumidity() {
        return humidity;
    }

    public String getStatus() {
        return status;
    }

    public double getStatusDouble() {
        return statusDouble;
    }

    public double getTemperature() {
        return temperature;
    }
}
