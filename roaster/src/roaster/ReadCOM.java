package roaster;

/**
 *
 * @author Matthieu Nahoum
 */
import gnu.io.CommPort;
import gnu.io.CommPortIdentifier;
import gnu.io.SerialPort;

import java.io.IOException;
import java.io.InputStream;

public class ReadCOM {

    public ReadCOM() {
    }

    public void connect(String portName) throws Exception {
        CommPortIdentifier portIdentifier = CommPortIdentifier.getPortIdentifier(portName);
        if (portIdentifier.isCurrentlyOwned()) {
            System.out.println("Error: Port is currently in use");
        } else {
            CommPort commPort = portIdentifier.open(this.getClass().getName(), 2000);

            if (commPort instanceof SerialPort) {
                SerialPort serialPort = (SerialPort) commPort;
                serialPort.setSerialPortParams(
                        57600,
                        SerialPort.DATABITS_8,
                        SerialPort.STOPBITS_1,
                        SerialPort.PARITY_NONE);
                InputStream in = serialPort.getInputStream();
                byte[] buffer = new byte[1024];
                int len = -1;
                try {
                    while ((len = in.read(buffer)) > -1) {
                        System.out.print(new String(buffer, 0, len));
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            } else {
                System.out.println("Error: Only serial ports are handled by this example.");
            }
        }
    }
}
