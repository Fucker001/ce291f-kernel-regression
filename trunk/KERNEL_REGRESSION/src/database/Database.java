
package database;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

/**
 *
 * @author Matthieu Nahoum
 */
public class Database {

    private String url;
    private String user;
    private String password;
    private Connection connection;



    public Database (String url, String user, String password) throws SQLException{
        this.url = url;
        this.user = user;
        this.password = password;
        this.connection = DriverManager.getConnection(this.url, this.user, this.password);
    }

    public ResultSet getResultSet(String sql) throws SQLException {

        Statement st = null;
        ResultSet rs = null;
        st = this.getConnection().createStatement(ResultSet.TYPE_SCROLL_INSENSITIVE, ResultSet.CONCUR_READ_ONLY);
        rs = st.executeQuery(sql);

        return rs;
    }

    public void close() throws SQLException {
        this.getConnection().close();
    }

    /**
     * @return the conn
     */
    public Connection getConnection() {
        return connection;
    }



}
