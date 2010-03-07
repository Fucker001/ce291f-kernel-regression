
package kernel;

import Jama.*;
import data.*;

/**
 *
 * @author Matthieu Nahoum
 */
public class Kernel {

    private DataSet dataset;
    private DataSet morphisedDataset;
    private Matrix input;
    private Matrix output;
    private Matrix kernel;
    private Matrix linearKernel;
    private Matrix gaussianKernel;
    private Matrix svdU;
    private Matrix svdS;
    private Matrix svdV;
    

    public Kernel(){
        this.input = this.dataset.getInput();
        this.output = this.dataset.getOutput();
    }

    private void computeKernel(double linearFactor, double gaussianFactor){
        this.kernel = this.gaussianKernel.times(gaussianFactor).plus(
                this.linearKernel.times(linearFactor));
        SingularValueDecomposition svd = this.kernel.svd();
    }

    private void computeLinearKernel(){
        int size = this.input.getColumnDimension();
        int dim = this.input.getRowDimension();
        this.linearKernel = new Matrix(size, size);
        for (int i = 0; i<size; i++){
            for (int j = 0; j<size; j++){
                double value = 0.0;
                for (int k = 0; k < dim; k++){
                    value += this.input.get(i,k) * this.input.get(j,k);
                }
                this.linearKernel.set(i,j,value);
            }
        }
    }

    private void computeGaussianKernel(double sigma){
        int size = this.input.getColumnDimension();
        int dim = this.input.getRowDimension();
        this.gaussianKernel = new Matrix(size, size);
        for (int i = 0; i<size; i++){
            for (int j = 0; j<size; j++){
                double value = 0.0;
                for (int k = 0; k < dim; k++){
                    double tmp = this.input.get(i,k) - this.input.get(j,k);
                    value += tmp*tmp;
                }
                value = java.lang.Math.exp(-value/(2.0 * sigma * sigma));
                this.gaussianKernel.set(i,j,value);
            }
        }
    }

    private void optimize(){
        Matrix modifiedInput = this.morphisedDataset.getModifiedInput();
        Matrix modifiedInputsTransposed = modifiedInput.transpose();
        Matrix theta = modifiedInputsTransposed.solve(this.output);

    }


}
