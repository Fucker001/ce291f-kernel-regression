
package csv;

import core.Time;

/**
 *
 * @author Matthieu Nahoum
 */
public class Match {

    private Time date;
    private double dateDouble;
    private double travelTime;
    private Weather weather;

    public Match(Time date, double travelTime, Weather weather){
        this.date = date;
        this.dateDouble = Match.formatDateOfDayInZeroOne(date);
        this.travelTime = travelTime;
        this.weather = weather;
    }

    public static double formatDateOfDayInZeroOne(Time date){
        int hour = date.get(Time.HOUR);
        int min = date.get(Time.MINUTE);
        int sec = date.get(Time.SECOND);
        final int totalSecInDay = 24 * 60 * 60;
        int total = hour * 3600 + min * 60 + sec;
        return ((double)total)/totalSecInDay;
    }

    public Time getDate() {
        return date;
    }

    public double getTravelTime() {
        return travelTime;
    }

    public Weather getWeather() {
        return weather;
    }

    public double getDateDouble() {
        return dateDouble;
    }

}
