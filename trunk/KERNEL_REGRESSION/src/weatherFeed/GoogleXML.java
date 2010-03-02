package weatherFeed;

import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;

/**
 *
 * @author Matthieu Nahoum
 */
public class GoogleXML {

    private HashMap<String, String> cities;

    public static void main(String[] args) throws Exception {
        new GoogleXML();
    }

    private GoogleXML() {
        this.cities = new HashMap<String, String>();
        this.cities.put("Berkeley, CA", "http://www.google.com/ig/api?weather=Berkeley,CA&hl=en");
        this.cities.put("San Francisco, CA", "http://www.google.com/ig/api?weather=San+Francisco,CA&hl=en");
        this.cities.put("Los Angeles, CA", "http://www.google.com/ig/api?weather=Los+Angeles,CA&hl=en");
        this.cities.put("Austin, TX", "http://www.google.com/ig/api?weather=Austin,TX&hl=en");
        this.cities.put("Chicago, IL", "http://www.google.com/ig/api?weather=Chicago,IL&hl=en");
        this.cities.put("Paris, IDF", "http://www.google.com/ig/api?weather=Paris,IDF&hl=en");
        try {
            for (String city : this.cities.keySet()) {
                this.parse(city, this.cities.get(city));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    /**
     * Parse the feed and get the weather data
     * @throws Exception
     */
    public void parse(String city, String stringURL) throws Exception {
        URL url = new URL(stringURL);
        DocumentBuilderFactory fabric = DocumentBuilderFactory.newInstance();
        DocumentBuilder constructor = fabric.newDocumentBuilder();
        Document document = constructor.parse(url.openStream());
        Node root = document.getDocumentElement();

        String condition = "";
        String temperature = "";
        String humidity = "";
        String wind = "";

        String windSpeed = "";
        String windDirection = "";

        ArrayList<Node> children = new ArrayList<Node>();
        for (int k = 0; k < root.getChildNodes().getLength(); k++) {
            children.add(root.getChildNodes().item(k));
        }
        for (Node child : children) {
            if (child.getNodeName().equals("weather")) {
                root = child;
            }
        }
        children = new ArrayList<Node>();
        for (int k = 0; k < root.getChildNodes().getLength(); k++) {
            children.add(root.getChildNodes().item(k));
        }
        for (Node child : children) {
            if (child.getNodeName().equals("current_conditions")) {
                root = child;
            }
        }
        children = new ArrayList<Node>();
        for (int k = 0; k < root.getChildNodes().getLength(); k++) {
            children.add(root.getChildNodes().item(k));
        }
        for (Node child : children) {
            if (child.getNodeName().equals("condition")) {
                NamedNodeMap attributes = child.getAttributes();
                for (int i = 0; i < attributes.getLength(); i++) {
                    if (attributes.item(i).getNodeName().equals("data")) {
                        condition = attributes.item(i).getNodeValue();
                    }
                }
            }
            if (child.getNodeName().equals("temp_f")) {
                NamedNodeMap attributes = child.getAttributes();
                for (int i = 0; i < attributes.getLength(); i++) {
                    if (attributes.item(i).getNodeName().equals("data")) {
                        temperature = attributes.item(i).getNodeValue();
                    }
                }
            }
            if (child.getNodeName().equals("humidity")) {
                NamedNodeMap attributes = child.getAttributes();
                for (int i = 0; i < attributes.getLength(); i++) {
                    if (attributes.item(i).getNodeName().equals("data")) {
                        humidity = attributes.item(i).getNodeValue();
                    }
                }
            }
            if (child.getNodeName().equals("wind_condition")) {
                NamedNodeMap attributes = child.getAttributes();
                for (int i = 0; i < attributes.getLength(); i++) {
                    if (attributes.item(i).getNodeName().equals("data")) {
                        wind = attributes.item(i).getNodeValue();
                        windDirection = wind.substring(6, wind.indexOf("at")-1);
                        windSpeed = wind.substring(wind.indexOf("mph")-2);
                    }
                }
            }
        }
        System.out.println("City:" + city
                + "\nCondition: " + condition
                + ", Temperature: " + temperature
                + " F, Humidity: " + humidity
                + ", Wind direction: " + windDirection
                + ", Wind speed: " + windSpeed + ".\n");
    }
}
