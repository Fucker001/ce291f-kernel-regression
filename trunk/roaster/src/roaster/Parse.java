package roaster;

import java.io.FileNotFoundException;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Properties;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.FloatControl;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.SourceDataLine;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 *
 * @author Matthieu Nahoum
 */
public class Parse {

    public static boolean takeAbsValues = false;
    public static double threshold = 100.0;
    public static String message = "No message";
    public static String soundFileName = "C:\\Users\\Matthieu\\Desktop\\tada.wav";
    public static String moteIdentifier = "$0001";
    public static FileWriter fw;
    public static ArrayList<Double> buffer;
    public static final int buffer_size = 50;
    public static double stdStop;

    /**
     * Test that the parser works properly
     * @param args
     */
    public static void main(String[] args) {
        double threshold = 20.3;
//        Parse.buffer = new ArrayList<Double>();
//        String lineTest1 = "$0001,-84.2F,2.6,0.17,N#";
//        System.out.println(Parse.parseAndAlert(lineTest1, threshold, true)); //true
//        System.out.println(Parse.parseAndAlert(lineTest1, threshold, false)); //false
//        System.out.println();
//        String lineTest2 = "$0001,-14.2F,2.6,0.17,N#";
//        System.out.println(Parse.parseAndAlert(lineTest2, threshold, true)); //false
//        System.out.println(Parse.parseAndAlert(lineTest2, threshold, false)); //false
//        System.out.println();
//        String lineTest3 = "$0001, 84.2F,2.6,0.17,N#";
//        System.out.println(Parse.parseAndAlert(lineTest3, threshold, true)); //true
//        System.out.println(Parse.parseAndAlert(lineTest3, threshold, false)); //true
//        System.out.println();
//        String lineTest4 = "$0001, 14.2F,2.6,0.17,N#";
//        System.out.println(Parse.parseAndAlert(lineTest4, threshold, true)); //false
//        System.out.println(Parse.parseAndAlert(lineTest4, threshold, false)); //falsetry
        try {
            Parse.ring();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void init(String propFile) throws IOException {
        Properties props = new Properties();
        props.load(new FileReader(new File(propFile)));
        Parse.soundFileName = props.getProperty("sound_file_name");
        Parse.takeAbsValues = new Boolean(props.getProperty("take_absolute_values"));
        Parse.threshold = new Double(props.getProperty("temperature_threshold"));
        Parse.moteIdentifier = props.getProperty("mote_identifier");
        Parse.fw = new FileWriter(new File(props.getProperty("file_to_write") + ".txt"));
        Parse.buffer = new ArrayList<Double>();
        Parse.stdStop = new Double(props.getProperty("stddev_threshold"));
        System.out.println("System initialized.");
    }

    /**
     * Parse a line and tell if the alarm should turn on
     * @param line The line coming from the COM port
     * @param threshold The threshold we want to respect
     * @param takeAbsValues True if we want to ignore the negative values and
     * consider them as positive.
     * @return true if the alarm should ring
     */
    public static boolean parseAndAlert(String line, double threshold, boolean takeAbsValues) {
        if (line.equals("") || !line.contains("F") || line == null) {
            System.err.println("Parse: invalid message.");
            return false;
        }
        if (!line.contains(Parse.moteIdentifier)) {
            System.out.println("---------" + line);
            return false;
        }
        System.out.println(line);
        boolean result = false;
        String[] intern = line.split(",");
        String mote = intern[0];
        String temp = intern[1];
        if (temp.contains("F")) {
            temp.replace("F", "");
        }
        if (temp.contains(" ")) {
            temp.replace(" ", "");
        }
        if (temp.contains("-") && takeAbsValues) {
            temp = temp.substring(1);
        }
        double temperature = Double.parseDouble(temp);
        double maximum = 0.0;
        double minimum = 2000.0;
        double avg = 0;
        double avg2 = 0;
        if (line.contains(Parse.moteIdentifier)) {
            //Store the temp in the buffer
            Parse.buffer.add(temperature);
            //Make sure the buffer is still size 50
            while(Parse.buffer.size()>Parse.buffer_size){
                Parse.buffer.remove(0);
            }
            // Compute the stddev and mean on the buffer
            for (double tBuf : Parse.buffer){
                if (tBuf > maximum){
                    maximum = tBuf;
                }
                if(tBuf < minimum){
                    minimum = tBuf;
                }
                avg+=tBuf;
                avg2+=tBuf*tBuf;
            }
            avg=avg/Parse.buffer.size();
            avg2=avg2/Parse.buffer.size();
            double stddev = java.lang.Math.sqrt(avg2-avg*avg);
            if (temperature > threshold) {
                result = true;
                Parse.message = "Temperature when stopped: " + temperature;
            }
            if(Parse.buffer.size() == Parse.buffer_size && stddev<Parse.stdStop){
                result = true;
                Parse.message = "Standard deviation when stopped: " + stddev;
            }
            System.out.println(temperature + "F....  Is it too high?  " + result);
            System.out.println("%%%%%%%%%%%%%%%%%%%%Stddev :" + stddev + "\n");
            Parse.storeData(mote, temperature);
        }
        return result;
    }

    public static void storeData(String identifier, double temperature) {
        String line = identifier + "," + temperature + "\n";
        try {
            Parse.fw.write(line);
            Parse.fw.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    
    // Play a sound
    private static final int EXTERNAL_BUFFER_SIZE = 524288; // 128Kb

    private enum Position {

        LEFT, RIGHT, NORMAL
    };
    private static Position curPosition = Position.NORMAL;

    /**
     * Ring the alarm
     */
    public static void ring() throws FileNotFoundException, IOException {
        File soundFile = new File(Parse.soundFileName);
        if (!soundFile.exists()) {
            System.err.println("Wave file not found: " + Parse.soundFileName);
            return;
        }

        AudioInputStream audioInputStream = null;
        try {
            audioInputStream = AudioSystem.getAudioInputStream(soundFile);
        } catch (UnsupportedAudioFileException e1) {
            e1.printStackTrace();
            return;
        } catch (IOException e1) {
            e1.printStackTrace();
            return;
        }

        AudioFormat format = audioInputStream.getFormat();
        SourceDataLine auline = null;
        DataLine.Info info = new DataLine.Info(SourceDataLine.class, format);

        try {
            auline = (SourceDataLine) AudioSystem.getLine(info);
            auline.open(format);
        } catch (LineUnavailableException e) {
            e.printStackTrace();
            return;
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        if (auline.isControlSupported(FloatControl.Type.PAN)) {
            FloatControl pan = (FloatControl) auline.getControl(FloatControl.Type.PAN);
            if (curPosition == Position.RIGHT) {
                pan.setValue(1.0f);
            } else if (curPosition == Position.LEFT) {
                pan.setValue(-1.0f);
            }
        }


        auline.start();

        int nBytesRead = 0;
        byte[] abData = new byte[EXTERNAL_BUFFER_SIZE];

        try {
            while (nBytesRead != -1) {
                nBytesRead = audioInputStream.read(abData, 0, abData.length);
                if (nBytesRead >= 0) {
                    auline.write(abData, 0, nBytesRead);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            return;
        } finally {
            auline.drain();
            auline.close();
        }
    }
}

