// Traffic analysis

//on Franklin, requires "module load acml"

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cstring>
#include <time.h>
#include <acml.h>
#include <string>
#include <sys/time.h>


//Constants for data sending
#define L1RESULT 0
//for blocking in calculations
#define BLOCKSIZE 32

#define EPSILON .0001


//
//  timer
//
double read_timer( ){
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

//Prints iDim x jDim matrix data
//Comes in column major, print in row
void printMat(double *matrix,unsigned int iDim,unsigned int jDim){
	for(unsigned int i=0;i<iDim;i++){
		for(unsigned int j=0;j<jDim;j++){
			printf("%f ",matrix[i+j*iDim]);
		}
		printf("\n");
	}
	printf("\n");
	return;
}

void calcSVD(double *in,double*out,unsigned int dim){
    // double simulation_time = read_timer( );
	
	//Calculates U*S^.5 where in = U*S*V^T, stores in out
	double *S = (double*)calloc(dim*dim,sizeof(double));
	double *U = (double*)calloc(dim*dim,sizeof(double));
	double *Vt = (double*)calloc(dim*dim,sizeof(double));
	int info=0;
	
	//DGESDD( JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK,LWORK, IWORK, INFO )
	dgesdd('A',dim,dim,in,dim,S,U,dim,Vt,dim,&info);
	
	/*
	
	//ORIGINAL
	//Multiply U*S^.5, store in out
	for(unsigned int i=0;i<dim;i++){
		for(unsigned int j=0;j<dim;j++){
			out[j+i*dim] = sqrt(S[i])*U[j+i*dim];
		}
	}
	*/
	
	//Blocked
	int blockSize = min(BLOCKSIZE,(int)dim);
	for(int y=0;y<((int)ceil(1.0*dim/blockSize));y++){
		for(int x=0;x<ceil(1.0*dim/blockSize);x++){
			for(int i=y*blockSize;i<y*blockSize+blockSize&&i<dim;i++){
				for(int j=x*blockSize;j<x*blockSize+blockSize&&j<dim;j++){
					out[j+i*dim] = sqrt(S[i])*U[j+i*dim];
				}
			}
		}			
	}
		
		
	//out now holds U*(S^.5)
	// ROW MAJOR i+j*dim
	// column major: j+i*dim
	
	//free calloc'ed space
	free(S);
	free(U);
	free(Vt);
	return;
}

void calcPolyKernel(double *in, double *out, unsigned int iDim, unsigned int jDim, double c1,double c2, double d){
	double *xi, *xj, xd[1], result;
	unsigned int maxDim = max(iDim,jDim);
	double *intermed = (double*)calloc(maxDim*maxDim,sizeof(double));
	double *intermed2 = (double*)calloc(maxDim*maxDim,sizeof(double));
	if (intermed==NULL||intermed2==NULL){
		printf("intermed calloc fail\n");
		return;
	}
	//get in^T to form column-major...
	/*
		//original
		
	for(unsigned int i=0;i<iDim;i++){
		for(unsigned int j=0;j<jDim;j++){
			intermed2[i+j*iDim] = in[j+i*jDim];
		}
	}
	*/
	
	//Blocked
	int blockSize = min(BLOCKSIZE,(int)min(jDim,iDim));
	for(int y=0;y<((int)ceil(1.0*iDim/blockSize));y++){
		for(int x=0;x<ceil(1.0*jDim/blockSize);x++){
			for(int i=y*blockSize;i<y*blockSize+blockSize&&i<iDim;i++){
				for(int j=x*blockSize;j<x*blockSize+blockSize&&j<jDim;j++){
					intermed2[i+j*iDim] = in[j+i*jDim];
					//out[j+i*dim] = sqrt(S[i])*U[j+i*dim];
				}
			}
		}			
	}
	
	//Form K[i,j]_(N,N)
	for(unsigned int i=0;i<jDim;i++){
		//get ith column
		xi = &(intermed2[i*iDim]);
		for(unsigned int j=0;j<jDim;j++){
			//get jth column
			xj = &(intermed2[j*iDim]);
			xd[0]=0;
			dgemm( 'T','N', 1,1,iDim, c2, xi,iDim, xj,iDim, 1, xd,1 );
			result = c1+xd[0];
			intermed[j+i*jDim] = pow(result,d);
		}
	}
	
	//Normalization
	double trace = 0;
	for(unsigned int i=0;i<maxDim;i++){
		trace += intermed[i+i*maxDim];	
	}
	double toD = 1/trace;
	//Scale by normalization factor
	dscal(maxDim*maxDim, toD, intermed,1);
	
	//compute SVD, store in out
	calcSVD(intermed,out,maxDim);
	
	//free calloc'ed space
	free(intermed);
	free(intermed2);
	return;
}
void calcGaussKernel(double *in,double *out,unsigned int iDim, unsigned int jDim,double sigma){
    double simulation_time = read_timer( );
	
	double *xi,*xj;
	double xd[iDim];
	double sum, result;
	unsigned int maxDim = max(iDim,jDim);
	double *intermed = (double*)calloc(maxDim*maxDim,sizeof(double));
	double *intermed2 = (double*)calloc(maxDim*maxDim,sizeof(double));
	
	/* //original
	//put array into column major
	for(unsigned int i=0;i<iDim;i++){
		for(unsigned int j=0;j<jDim;j++){
			intermed2[i+j*iDim] = in[j+i*jDim];
		}
	}
	*/
	//Blocked
	int blockSize = min(BLOCKSIZE,(int)min(jDim,iDim));
	for(int y=0;y<((int)ceil(1.0*iDim/blockSize));y++){
		for(int x=0;x<ceil(1.0*jDim/blockSize);x++){
			for(int i=y*blockSize;i<y*blockSize+blockSize&&i<iDim;i++){
				for(int j=x*blockSize;j<x*blockSize+blockSize&&j<jDim;j++){
					intermed2[i+j*iDim] = in[j+i*jDim];
					//out[j+i*dim] = sqrt(S[i])*U[j+i*dim];
				}
			}
		}			
	}
	//array in column major
	for(unsigned int i=0;i<jDim;i++){
		//get ith column
		xi = &(intermed2[i*iDim]);
		for(unsigned int j=0;j<jDim;j++){
			//get jth column
			xj = &(intermed2[j*iDim]);
			for(unsigned int q=0;q<iDim;q++){
				//for each item, find difference
				xd[q] = xi[q]-xj[q];
			}
			//sum items in column + norm it
			sum = 0;
			for(unsigned int q=0;q<iDim;q++){
				sum += pow(abs(xd[q]),2);
			}
			//square root sum
			result = sqrt(sum);
			//square to normalize
			intermed[j+i*jDim] = pow(result,2);
		}
	}
	
	//need exponential... of intermed
	for(unsigned int i=0;i<maxDim*maxDim;i++){
		intermed[i] = exp(-1*(intermed[i]/(2*pow(sigma,2))));
	}
	
	//Normalization
	double trace = 0;
	for(unsigned int i=0;i<maxDim;i++){
		trace += intermed[i+i*maxDim];	
	}
	double toD = 1/trace;
	dscal(maxDim*maxDim, toD, intermed,1);
	
	//compute SVD, store in out
	calcSVD(intermed,out,maxDim);
	
	free(intermed);
	free(intermed2);
	return;
}

void calcLinKernel(double *in,double *out,unsigned int iDim,unsigned int jDim){
	unsigned int maxDim = max(iDim,jDim);
	unsigned int minDim = min(iDim,jDim);
	double *intermed = (double*)calloc(maxDim*maxDim,sizeof(double));
	if (intermed==NULL){
		printf("intermed calloc fail\n");
		return;
	}
	//multiply X'*X, X already in row major, X' is column	
	dgemm('N', 'T', maxDim, maxDim, minDim, 1, in, maxDim, in, maxDim, 1, intermed, maxDim);
	
	//Normalization
	double trace = 0;
	for(unsigned int i=0;i<maxDim;i++){
		trace += intermed[i+i*maxDim];	
	}
	double toD = 1/trace;
	//Scale by normalization factor
	dscal(maxDim*maxDim, toD, intermed,1);
	
	// compute SVD, store in out
	calcSVD(intermed,out,maxDim);
	
	//free calloc'ed space
	free(intermed);
	return;

}
//Makes sure option is found
int find_option( int argc, char **argv, const char *option ){
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}
//Grabs data from commandline args
char *read_string( int argc, char **argv, const char *option, char *default_value ){
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}
//grab int from command line...
int read_int( int argc, char **argv, const char *option, int default_value ){
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}
//Prints correct usage info
void printUsage(){
	printf( "Options:\n" );
	printf( "-h to see this help\n" );
	printf( "-i <file> Input CSV file with input characteristics\n" );
	printf( "-o <file> Observed travel time CSV file \n" );
	printf( "-f <file> Kernel configuration\n" );
}
//Gets the number of values separated by delim (how many values in csv line)
unsigned int getNumIn(char *record, char *delim){
    char*p=strtok(record,delim);
    unsigned int fld=0;
    while(p){
		fld++;
		p=strtok('\0',delim);
	}		
	return fld;
}
//Parses csv line (record) using delim as separator. Stores each one in row arr
void parse( char *record, char *delim, double *arr){
    char *p=strtok(record,delim);
	double t;
	unsigned int fld=0;
    while(p){
		t = atof(p);
        memcpy(&(arr[fld]),&t,sizeof(double));
		p=strtok('\0',delim);
		fld++;
	}		
}
//Parses kernel.csv for kernel config...
/* format
G,sigma
L
P,c1,c2,d

*/
void parseKernel( char *record, char *delim, int *numG, int *numL, int *numP, double *sigma,double *c1,double *c2,double *d){
    char *p=strtok(record,delim);
	double t;
	// printf("__Found char: %c %s\n",p[0],p);
	if (p[0]=='G'){
		//gaussian input
		p=strtok('\0',delim);
		t = atof(p);
		memcpy(&(sigma[*numG]),&t,sizeof(double));
		(*numG)++;
	}else if(p[0]=='L'){
		//Linear input
		(*numL)++;
	}else if(p[0]=='P'){
		//Poly input
		p=strtok('\0',delim);
		t = atof(p);
		memcpy(&(c1[*numP]),&t,sizeof(double));
		p=strtok('\0',delim);
		t = atof(p);
		memcpy(&(c2[*numP]),&t,sizeof(double));
		p=strtok('\0',delim);
		t = atof(p);
		memcpy(&(d[*numP]),&t,sizeof(double));
		(*numP)++;
	}else{
		printf("Invalid kernel csv\n");
		exit(1);
	}
	while(p){
		p=strtok('\0',delim);
	}
}
//only gets kernel counts so far
void parseKernelInit( char *record, char *delim, int *numG, int *numL, int *numP){
    char *p=strtok(record,delim);
	// printf("++Found char: %c %s\n",p[0],p);
	if (p[0]=='G'){
		//gaussian input
		(*numG)++;
	}else if(p[0]=='L'){
		//Linear input
		(*numL)++;
	}else if(p[0]=='P'){
		//Poly input
		(*numP)++;
	}else{
		printf("Invalid kernel csv\n");
		exit(1);
	}
	while(p){
		p=strtok('\0',delim);
	}
}

void getBeta(double *Beta,double *u, int Np,double *cMajor,double *rMajor,double *output,int Nt,int dim){
	double *KtU = (double*) calloc(Nt,sizeof(double));
	//matrix*vector to calculate right hand side, once only
	dgemv('N',Nt,Np,1.0,rMajor,Nt,u,1,1.0,KtU,1);
	for(int i=0;i<Nt;i++){
		KtU[i] = output[i]-KtU[i];
	}
	// KtU is now the right hand side. 
	for(int i=0;i<Np;i++){
		double temp = 0;
		for(int j=0;j<Nt;j++){
			temp += -1*cMajor[j+i*dim]*KtU[j];
		}
		Beta[i] = temp;
	}
	free(KtU);
}
void getAlpha(double *Alpha,double *u, int Np,double *cMajor,double *rMajor,double *output,int Nt,int dim){
	double *KtU = (double*) calloc(Nt,sizeof(double));
	//matrix*vector to calculate right hand side, once only
	dgemv('N',Nt,Np,1.0,rMajor,Nt,u,1,1.0,KtU,1);
	double temp=0;
	for(int i=0;i<Nt;i++){
		// KtU[i] = output[i]-KtU[i];
		temp += pow(output[i]-KtU[i],2);
	}
	*Alpha = sqrt(temp);
	free(KtU);
}
//calculates the phi function
double phiOfU(double *U,int Np,double *cMajor,double *rMajor,double *output,int Nt,int dim){
	double toRet=0;
	for(int i=0;i<Np;i++){
		toRet += abs(U[i]);
	}
	double rightHandSide;
	getAlpha(&rightHandSide,U,Np,cMajor,rMajor,output,Nt,dim);
	return toRet + rightHandSide;
}

//run l1 optimization
void calcL1(double *u,int Np,double *cMajor,double *rMajor,double *output,int Nt,int dim){
	//initial U vector gets passed in....

	double *Beta=(double*) calloc(Np,sizeof(double));
	double Alpha;
	//Hold the gradient vector
	double *Gradient=(double*) calloc(Np,sizeof(double));
	//Hold the derivative of u
	double *uPrime=(double*) calloc(Np,sizeof(double));
	// will be used to shift u's
	// double *tempU;
	
	int stepsToTake=100;
	while(stepsToTake > 0){
		getBeta(Beta,u,Np,cMajor,rMajor,output,Nt,dim);
		getAlpha(&Alpha,u,Np,cMajor,rMajor,output,Nt,dim);
		for(int i=0;i<Np;i++){
			Gradient[i] = (u[i]<0?-1:1) + Beta[i]/Alpha;
		}
		//differentiate U for next iteration
		
		// iterate columns
		for(int i=0;i<Np;i++){
			double sum =0;
			// iterate rows
			for(int j=0;j<Nt;j++){
				sum += pow(cMajor[j+i*dim],2);
			}
			uPrime[i] = sum/Alpha + pow(Beta[i],2)/pow(Alpha,3);
		}
		
		// Newton's method
		for(int i=0;i<Np;i++){
			// top part of newtone's branch
			if (uPrime[i] != 0){
				u[i] = u[i] - (Gradient[i]/uPrime[i]);
			}else{
				// u[i] = u[i];
			}
		}
		// Now u holds the next u for the iteration
		// check if phi(u) is now <= epsilon...
		if (phiOfU(u,Np,cMajor,rMajor,output,Nt,dim) <= EPSILON){
			break;
		}
		stepsToTake--;
	}
	// u holds the optimal u
	free(Beta);
	free(Gradient);
	return;
}
//get parameter index for a given kernel
int paramInd(unsigned int *order,int ind){
	int n=0;
	for(int i=0;i<ind;i++){
		if (order[ind] == order[i])
			n++;
	}
	return n;
}

int main( int argc, char **argv ){ 

	
    double simulation_time = read_timer( );
	
	

	if( find_option( argc, argv, "-h" ) >= 0 ){
		printUsage();
        return 0;
    }
	//get input/output file names
	char *inputFile = read_string( argc, argv, "-i", NULL );
	char *outputFile = read_string( argc, argv, "-o", NULL );
	char *kernelFile = read_string( argc, argv, "-f", NULL );
	
	if (inputFile == NULL || outputFile == NULL || kernelFile==NULL){
		printUsage();
        exit(1);
	}
	
	//Set up parallel cores using MPI
	int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	
	unsigned int numInputPoints;
	unsigned int dimension=0;
	
	int numGaussian=0;
	int numLinear=0;
	int numPoly=0;
	
	// if (rank==0){
		// printf("Beginning time:\t%f\n",read_timer( )-simulation_time);
	// }
	if (rank==0){
		//only host core does this, get file info (#dimensions,#input points)
		printf("In: %s, out %s\n",inputFile,outputFile);
		unsigned int tempSize = 10000;
		char *temp = (char*)calloc(tempSize , sizeof(char));
		//figure out the # of input points
		FILE *fi = fopen(inputFile,"r");
		fgets(temp,tempSize,fi);
		dimension++;
		numInputPoints = getNumIn(temp,",");
		while(fgets(temp,tempSize,fi)){
			dimension++;
		}
		fclose(fi);
		
		fi = fopen(kernelFile,"r");
		//Get info on kernels...
		while(fgets(temp,tempSize,fi)){
			parseKernelInit( temp, ",",&numGaussian,&numLinear,&numPoly);
		}
		// printf("RANK0: G:%d, L:%d, P:%d\n",numGaussian,numLinear,numPoly);
		fclose(fi);
		free(temp);
	}
	
	//Send dimension/numInputPoints to all cores
	MPI_Bcast(&numInputPoints,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	MPI_Bcast(&dimension,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	MPI_Bcast(&numGaussian,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	MPI_Bcast(&numLinear,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	MPI_Bcast(&numPoly,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	// printf("R:%d G:%d, L:%d, P:%d\n",rank, numGaussian,numLinear,numPoly);
	// printf("d\n");
	// fflush(stdout);
	double *gaussianParam = (double*)calloc(numGaussian,sizeof(double));
	double *polyC1 = (double*)calloc(numPoly,sizeof(double));
	double *polyC2 = (double*)calloc(numPoly,sizeof(double));
	double *polyD = (double*)calloc(numPoly,sizeof(double));
	
	
	//get max of dimension,#input points. Will be dimension of kernels...
	unsigned int maxDim = max(numInputPoints,dimension);
	unsigned int maxDimTotalLength = maxDim*maxDim;
	unsigned int rowsInTraining = ceil(maxDim/2);
	unsigned int rowsInTest = maxDim-rowsInTraining;
	//Check that all cores got them. Correct.
	// printf("R:%d numInputPoints:%d, dimension:%d\n",rank,numInputPoints,dimension);
	
	// if (rank==0){
		// printf("Halfway done reading files:\t%f\n",read_timer( )-simulation_time);
	// }
	//Make space for the input data
	double *input = (double*) calloc(dimension*numInputPoints,sizeof(double));
	if (input == NULL){
		//make sure space was created
		printf("R:%d Error calloc'ing on input %d bytes\n",rank,dimension*numInputPoints);
		exit(1);
	}
	//make space for output data
	double *output = (double*) calloc(numInputPoints,sizeof(double));
	if (output == NULL){
		printf("R:%d Error calloc'ing on output\n",rank);
		exit(1);
	}
	
	if(rank == 0){
		//host cpu grabs data from files.
		
		//parse csv files
		unsigned int tempSize = 10000;
		char *temp = (char*)calloc(tempSize , sizeof(char));
		
		//Grab input csv
		FILE *fi = fopen(inputFile,"r");
		unsigned int c=0;
		while(fgets(temp,tempSize,fi)){
			parse( temp, ",",&(input[c*numInputPoints]));
			c++;
		}
		fclose(fi);
		
		//grab output csv
		FILE *fo = fopen(outputFile,"r");
		fgets(temp,tempSize,fo);
		parse(temp, ",",output);
		fclose(fo);
		// printMat(output,1,numInputPoints);
		
		fi = fopen(kernelFile,"r");
		//Get info on kernels...
		
		// reset for next step
		numGaussian=0;
		numLinear=0;
		numPoly=0;
		
		while(fgets(temp,tempSize,fi)){
			parseKernel( temp, ",",&numGaussian,&numLinear,&numPoly,gaussianParam,polyC1,polyC2,polyD);
		}
		
		// printf("R:%d G:%d(%f) L:%d P:%d(%f,%f,%f)\n",rank,numGaussian,gaussianParam[0],numLinear,
	// numPoly,polyC1[0],polyC2[0],polyD[0]);
		fclose(fi);
		
		free(temp);
	}
	//broadcast input data to all cores : correct
	// printf("R:%d waiting/sending G:%d, P:%d\n",rank, numGaussian,numPoly);
	MPI_Bcast(input,dimension*numInputPoints,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(gaussianParam,numGaussian,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(polyC1,numPoly,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(polyC2,numPoly,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(polyD,numPoly,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	unsigned int numKernels = numGaussian+numLinear+numPoly;
	// unsigned int fatKColumns = maxDim*numKernels;
	// printf("R:%d G:%d(%f) L:%d P:%d(%f,%f,%f)\n",rank,numGaussian,gaussianParam[0],numLinear,
	// numPoly,polyC1[0],polyC2[0],polyD[0]);
	
	// MPI_Barrier(MPI_COMM_WORLD);
	// return 0;
	
	if (rank==0){
		printf("Set up time:\t%f\n",read_timer( )-simulation_time);
	}
	
	unsigned int kernAss[numKernels];
	for(unsigned int i=0;i<numGaussian;i++){
		kernAss[i] = 0;
	}
	for(unsigned int i=0;i<numLinear;i++){
		kernAss[i+numGaussian] = 1;
	}
	for(unsigned int i=0;i<numPoly;i++){
		kernAss[i+numGaussian+numLinear] = 2;
	}
	//get kernels for this rank to compute...
	//shuffle the job order
	srand(n_proc);
	unsigned int t,j;
	for(unsigned int i=0;i<numKernels-1;i++){
		j = rand()%numKernels;
		t = kernAss[j];
		kernAss[j] = kernAss[i];
		kernAss[i] = t;
	}
	//allocate space for most this proc would have to work on...
	int procWorkSize[n_proc];
	int recvOffset[n_proc];
	for(int i=0;i<n_proc;i++){
		procWorkSize[i] = (int)ceil(1.0*(numKernels-i)/n_proc)*maxDimTotalLength;
		if (i>0)
			recvOffset[i] = recvOffset[i-1]+procWorkSize[i-1];
		else
			recvOffset[i] = 0;
	}
	double *workSpace = (double*)calloc(procWorkSize[rank],sizeof(double));
	int workedSoFar = 0;
	int paramIndex;
	for(unsigned int i=rank;i<numKernels;i+=n_proc){
		paramIndex = paramInd(kernAss,i);
		if(kernAss[i] == 0){
			// printf("R:%d calculating gaussian %d\n",rank,i);
			calcGaussKernel(input,workSpace+(workedSoFar*maxDimTotalLength),dimension,numInputPoints,gaussianParam[paramIndex]);
		}else if(kernAss[i] == 1){
			// printf("R:%d calculating linear %d\n",rank,i);
			calcLinKernel(input,workSpace+(workedSoFar*maxDimTotalLength),dimension,numInputPoints);
		}else if(kernAss[i] == 2){
			// printf("R:%d calculating poly %d\n",rank,i);
			calcPolyKernel(input, workSpace+(workedSoFar*maxDimTotalLength),dimension,numInputPoints, polyC1[paramIndex],polyC2[paramIndex], polyD[paramIndex]);
		}
		workedSoFar++;
	}
// printf("R:%d workLen:%d proc todo:%d done:%d sending #s:%d at offset:%d\n",
// rank,procWorkSize[rank],procWorkSize[rank]/maxDimTotalLength,workedSoFar,workedSoFar*maxDimTotalLength,recvOffset[rank]);
	
	double *fatKColumnMajor,*fatKRowMajor;//,*Ktraining;
	// if(rank==0){
		fatKColumnMajor = (double*)calloc(maxDimTotalLength*numKernels,sizeof(double));
		fatKRowMajor = (double*)calloc(maxDimTotalLength*numKernels,sizeof(double));
	// }
	
	if (rank==0){
		printf("Calculation time:\t%f\n",read_timer( )-simulation_time);
	}
	
	//gather the kernels back to the main.
	// MPI_Gatherv ( workSpace,workedSoFar*maxDimTotalLength , MPI_DOUBLE, fatKColumnMajor, procWorkSize, recvOffset, MPI_DOUBLE, 0, MPI_COMM_WORLD );
	MPI_Allgatherv ( workSpace,workedSoFar*maxDimTotalLength , MPI_DOUBLE, fatKColumnMajor, procWorkSize, recvOffset, MPI_DOUBLE, MPI_COMM_WORLD );
		
	
	//Fat k is now in column major. 
	// easiest to bring together in column, then transpose. easier than interleaving portions...
	if (rank==0){
		printf("Gather time:\t%f\n",read_timer( )-simulation_time);
	}
	
	
	free(workSpace);
	
	//Throw fat k into row major to establish training/test vectors
	// Blocked tranpose
	int blockSize = min(BLOCKSIZE,(int)maxDim);
	for(int y=0;y<((int)ceil(1.0*maxDim/blockSize));y++){
		for(int x=0;x<ceil(1.0*maxDim*numKernels/blockSize);x++){
			for(int i=y*blockSize;i<y*blockSize+blockSize&&i<maxDim;i++){
				for(int j=x*blockSize;j<x*blockSize+blockSize&&j<maxDim*numKernels;j++){
					fatKRowMajor[j+i*maxDim*numKernels] = fatKColumnMajor[i+j*maxDim];
				}
			}
		}
	}		
		
	if (rank==0){
		printf("reformat fatKColumnMajor:\t%f\n",read_timer( )-simulation_time);
	}
	// if (rank==2){
		// printf("\nRank:%d fatK:\n",rank);
		// printMat(fatKColumnMajor,maxDim,maxDim*numKernels);
		// printf("\n\n");

		// printf("\nRank:%d fatKRowMajor:\n",rank);
		// printMat(fatKRowMajor,maxDim*numKernels,maxDim);
		// printf("\n\n");
	// }
	
	
	//Calculate own starting location...
	double *U = (double*) calloc (numKernels*maxDim,sizeof(double));
	for(int i=0;i<numKernels*maxDim;i++){
		U[i] = rand();
	}
	// have each slave proc calculate and send back result
	if (rank != 0){
		//optimize L1
		calcL1(U,numKernels*maxDim,fatKColumnMajor,fatKRowMajor,output,rowsInTraining,maxDim);
		// now U holds the optimal u
		MPI_Send( U, numKernels*maxDim, MPI_DOUBLE, 0, L1RESULT , MPI_COMM_WORLD );
	}else{
		MPI_Request result;
		for(int i=1;i<n_proc;i++){
			MPI_Irecv( U,numKernels*maxDim, MPI_DOUBLE, i, L1RESULT , MPI_COMM_WORLD, &result );
		}
		
		//now wait for a result to be returned.
		MPI_Status status;
		MPI_Wait(&result,&status);
		
		double vStar = 0;
		double *KtU = (double*) calloc(rowsInTraining,sizeof(double));
		//matrix*vector to calculate right hand side, once only
		dgemv('N',rowsInTraining,numKernels*maxDim,1.0,fatKRowMajor,rowsInTraining,U,1,1.0,KtU,1);
		for(int i=0;i<rowsInTraining;i++){
			// KtU[i] = output[i]-KtU[i];
			vStar += pow(output[i]-KtU[i],2);
		}
		vStar = pow(sqrt(vStar),2);
		double phiStar = phiOfU(U,numKernels*maxDim,fatKColumnMajor,fatKRowMajor,output,rowsInTraining,maxDim);
		double muStar = vStar/phiStar;
		
		// build eigenvalues
		double *Lambdas = (double*) calloc(numKernels*maxDim,sizeof(double));
		for(int i=0;i<numKernels*maxDim;i++){
			Lambdas[i] = abs(U[i])/phiStar;
		}
		
		double *KStar = (double*) calloc(maxDimTotalLength,sizeof(double));
		
		for(int i=0;i<numKernels*maxDim;i++){
			double *KColumn = fatKColumnMajor+i*maxDim;
			// multiply EV*Trans(EV) and scale it by Lambdas[i]
			// by putting KStar as C parameter, we add the product to it each time.
			// alpha*op( A )*op( B ) + beta*C
			dgemm('N','T',maxDim,maxDim,1,Lambdas[i],KColumn,maxDim,KColumn,maxDim,1.0,KStar,maxDim);
		}
		
		// form ktraining
		double *KTraining = (double*) calloc(rowsInTraining*rowsInTraining,sizeof(double));
		double *KTrainingTemp = (double*) calloc(rowsInTraining*rowsInTraining,sizeof(double));
		for(int i=0;i<rowsInTraining;i++){
			for(int j=0;j<rowsInTraining;j++){
				KTraining[i+j*rowsInTraining] = KStar[i+j*maxDim];
				KTrainingTemp[i+j*rowsInTraining] = KTrainingTemp[i+j*maxDim];
			}
		}
		// add muStar to KTraining
		for(int i=0;i<rowsInTraining;i++){
			KTraining[i+i*rowsInTraining] += muStar;
			KTrainingTemp[i+i*rowsInTraining] += muStar;
		}
		// KTraining is now formed
		
		// form X*, of size (N-Nt)xNt
		double *XStar = (double*) calloc((maxDim-rowsInTraining)*(rowsInTraining),sizeof(double));
		for(int i=rowsInTraining;i<maxDim;i++){
			for(int j=0;j<rowsInTraining;j++){
				XStar[i+j*(maxDim-rowsInTraining)] = KStar[i+j*maxDim];
			}
		}
		
		// form a*, KTraining <- inv(KTraining)
		// get pivot information
		int *pivotInfo = (int*)calloc(rowsInTraining,sizeof(int));
		int success=0;
		dgetrf( rowsInTraining, rowsInTraining, KTrainingTemp, rowsInTraining, pivotInfo, &success );
		if (success != 0){
			printf("DGETRF failed?\n");
		}
		//Do the inverse computation, store in KTraining
		dgetri( rowsInTraining,KTraining,rowsInTraining,pivotInfo,&success);
		if (success != 0){
			printf("DGETRI failed?\n");
		}
		double *aStar = (double*)calloc(rowsInTraining,sizeof(double));
		dgemv('N',rowsInTraining,rowsInTraining,1.0,KTraining,rowsInTraining,output,1,1.0,aStar,1);
		
		// actually do estimation
		double *estimate = (double*)calloc(maxDim-rowsInTraining,sizeof(double));
		dgemv('N',maxDim-rowsInTraining,rowsInTraining,1.0,XStar,maxDim-rowsInTraining,aStar,1,1.0,estimate,1);
		
		// printf("Estimates:\n");
		// for(int i=0;i<maxDim-rowsInTraining;i++){
			// printf("%f ",estimate[i]);
		// }
		// printf("\n");
		
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
    if( rank == 0 ){
        printf("input:%dx%d n_procs = %d, simulation time = %g s\n", dimension,numInputPoints, n_proc, read_timer( )-simulation_time );
		printf("NumKernels: %d\n",numKernels);	
		printf("Linear: used:%d\n",numLinear);
		printf("Polynomial: used:%d\tc1:%f,c2:%f,d:%f\n",numPoly,polyC1[0],polyC2[0],polyD[0]);
		printf("Gaussian: used:%d\tsigma:%f\n",numGaussian,gaussianParam[0]);
		printf("fatKColumnMajor: %u values, %u bytes\n",maxDimTotalLength*numKernels,maxDimTotalLength*numKernels*sizeof(double));
		printf("Training: %d rows, Testing: %d rows\n",rowsInTraining,rowsInTest);
    }
	
	//Free all calloc'ed space
	// if (rank == 0){
		free(fatKColumnMajor);
		free(fatKRowMajor);
	// }
	free(input);
	free(output);
	// free(Ktraining);
	//For testing purposes, have all cores finish together
	MPI_Barrier(MPI_COMM_WORLD);
	
	// printf("R:%d Finishing application\n",rank);
	// fflush(stdout);
	return 0;
}
