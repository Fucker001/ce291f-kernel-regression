package weatherFeed;

import java.io.StringReader;
import java.net.URL;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.xml.sax.InputSource;

/**
 *
 * @author Matthieu Nahoum
 */
public class GoogleXML {

    public static void main(String[] args){
        
    }

    public void parse(String xml) throws Exception {
        URL url = new URL("http://www.google.com/ig/api?weather=Berkeley,CA&hl=en");

        DocumentBuilderFactory fabric = DocumentBuilderFactory.newInstance();
        DocumentBuilder constructor = fabric.newDocumentBuilder();
        Document document = constructor.parse(new InputSource(new StringReader(xml)));
        Node root = document.getDocumentElement();


    }
}
