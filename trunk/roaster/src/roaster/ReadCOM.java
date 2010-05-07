package roaster;

/**
 *
 * @author Matthieu Nahoum
 */
import gnu.io.CommPort;
import gnu.io.CommPortIdentifier;
import gnu.io.NoSuchPortException;
import gnu.io.PortInUseException;
import gnu.io.SerialPort;
import gnu.io.UnsupportedCommOperationException;

import java.io.IOException;
import java.io.InputStream;

public class ReadCOM {

    public ReadCOM() {
    }

    public void connect(String portName)
            throws AlarmThrownException,
            NoSuchPortException,
            PortInUseException,
            UnsupportedCommOperationException,
            IOException {
        CommPortIdentifier portIdentifier =
                CommPortIdentifier.getPortIdentifier(portName);
        CommPort commPort =
                portIdentifier.open(this.getClass().getName(), 2000);
        SerialPort serialPort = (SerialPort) commPort;
        serialPort.setSerialPortParams(
                57600,
                SerialPort.DATABITS_8,
                SerialPort.STOPBITS_1,
                SerialPort.PARITY_NONE);
        InputStream in = serialPort.getInputStream();
        System.out.println("Input stream caught.");
        byte[] buffer = new byte[1024];
        int len = -1;
        String line = "";
        while ((len = in.read(buffer)) > -1) {
            line += new String(buffer, 0, len);
            if (line.contains("\r\n")) {
                String[] coco = line.split("\r\n");
                if (coco.length >= 1) {
                    for (int i = 0; i < coco.length; i++) {
                        // Test if the alarm should be thrown
                        boolean alert = Parse.parseAndAlert(
                                coco[i],
                                Parse.threshold,
                                Parse.takeAbsValues);
                        if (alert) {
                            throw new AlarmThrownException(Parse.message);
                        }

                    }
                    if (coco.length >= 2) {
                        line = coco[coco.length - 1];
                    } else {
                        line = "";
                    }
                } else {
                    line = "";
                }
            }



        }
    }
}
