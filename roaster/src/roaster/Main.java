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
            ReadCOM rcom = new ReadCOM();
            rcom.connect("COM5");
        }catch (AlarmThrownException alarm){
            try {
                Parse.ring();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
