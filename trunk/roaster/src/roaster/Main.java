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
            ReadCOM rcom = new ReadCOM();
            rcom.connect("COM5");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
