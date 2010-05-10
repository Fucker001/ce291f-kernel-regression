package roaster;

/**
 *
 * @author Matthieu Nahoum
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try {
            Parse.init(args[0]);
//            Parse.init("./src/roaster/roaster.properties");
            ReadCOM rcom = new ReadCOM();
            rcom.connect(args[1]);
        }catch (AlarmThrownException alarm){
            try {
                // Wait 10 sec
                Thread.sleep(10000);
                // Ring
                Parse.ring();
                // Wait 25 secs
                Thread.sleep(24000);
                // Countdown
                Parse.soundFileName = Parse.countdownFileName;
                Parse.ring();
                System.exit(0);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
