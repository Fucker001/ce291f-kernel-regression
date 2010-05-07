package roaster;

/**
 *
 * @author Matthieu Nahoum
 */
public class AlarmThrownException extends Exception{

    public AlarmThrownException(String message){
        super(message);
        System.err.println(message);
    }

}
