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
            //Parse.init("./src/roaster/roaster.properties");
            ReadCOM rcom = new ReadCOM();
            rcom.connect("COM5");
        }catch (AlarmThrownException alarm){
            try {
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
