package weatherFeed;

import java.io.InputStream;
import java.net.URL;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpression;
import javax.xml.xpath.XPathExpressionException;
import javax.xml.xpath.XPathFactory;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

/**
 * Thanks to Developpez.com
 * @author Matthieu Nahoum
 */
public class Feed {

    public static NodeList eval(InputStream stream, String expression) {
        NodeList list = null;
        try {
            //create the source
            InputSource source = new InputSource(stream);

            //create the XPath
            XPathFactory fabric = XPathFactory.newInstance();
            XPath xpath = fabric.newXPath();

            //evaluation of XPath expression
            XPathExpression exp = xpath.compile(expression);
            list = (NodeList) exp.evaluate(source, XPathConstants.NODESET);

        } catch (XPathExpressionException xpee) {
            xpee.printStackTrace();
        }
        return list;
    }

    public static void main(String[] args) {
        try {
            URL url = new URL("http://weather.yahooapis.com/forecastrss?w=2362930&u=c");

            String expression = "//item/title";

            System.out.println("Results:\n\n");
            NodeList liste = eval(url.openStream(), expression);
            if (liste != null) {
                for (int i = 0; i < liste.getLength(); i++) {
                    Node node = liste.item(i);
                    System.out.println(node.getTextContent());
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
