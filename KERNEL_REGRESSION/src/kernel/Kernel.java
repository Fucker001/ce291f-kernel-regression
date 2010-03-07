
package kernel;

import data.Input;
import Jama.*;
import data.Output;

/**
 *
 * @author Matthieu Nahoum
 */
public class Kernel {

    private Input inputTraining;
    private Input inputValidation;
    private Input input;
    private Output outputTraining;
    private Output outputValidation;
    private Output output;
    private Matrix kernel;
    private Matrix linearKernel;
    private Matrix gaussianKernel;
    private Matrix svdU;
    private Matrix svdS;
    private Matrix svdV;
    

    public Kernel(){

    }

    private void computeLinearKernel(){
        int size = this.input.getNumInputs();
        int dim = this.input.getDimension();
        this.linearKernel = new Matrix(size, size);
        for (int i = 0; i<size; i++){
            for (int j = 0; j<size; j++){
                double value = 0.0;
                for (int k = 0; k < dim; k++){
                    value += this.input.getElement(i,k) * this.input.getElement(j,k);
                }
                this.linearKernel.set(i,j,value);
            }
        }
    }

    private void computeGaussianKernel(double sigma){
        int size = this.input.getNumInputs();
        int dim = this.input.getDimension();
        this.linearKernel = new Matrix(size, size);
        for (int i = 0; i<size; i++){
            for (int j = 0; j<size; j++){
                double value = 0.0;
                for (int k = 0; k < dim; k++){
                    double tmp = this.input.getElement(i,k) - this.input.getElement(j,k);
                    value += tmp*tmp;
                }
                value = java.lang.Math.exp(-value/(2.0 * sigma * sigma));
                this.linearKernel.set(i,j,value);
            }
        }
    }



}
