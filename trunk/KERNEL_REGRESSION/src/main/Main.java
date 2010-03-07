/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import data.DataSet;
import kernel.*;

/**
 *
 * @author matthieu
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        DataSet dataset = new DataSet();

        Kernel ksolve = new Kernel(dataset);
    }
}
