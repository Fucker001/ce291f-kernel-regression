package kernel;

import Jama.*;
import data.DataSet;

/**
 *
 * @author Matthieu Nahoum
 */
public class L1RegLSESolver {

    private Matrix K1;
    private Matrix output;
    private double phiStar;
    private Matrix uStar;

    public L1RegLSESolver(Matrix K1, DataSet dataset) {
        this.K1 = K1;
        this.output = dataset.getOutput().getMatrix(0, 0, 0, dataset.getOutput().getColumnDimension() / 2 - 1);
    }

    public void solve() {
        
    }

    public double getPhiStar() {
        return phiStar;
    }

    public Matrix getuStar() {
        return uStar;
    }
}
