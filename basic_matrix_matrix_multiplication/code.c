#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
			       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
	int posx = blockIdx.x*blockDim.x + threadIdx.x;
	int posy = blockIdx.y*blockDim.y + threadIdx.y;
	int cpos = posy*numCColumns+posx;
	int k;
	float sum=0.0;
	
	if ((posx<numCColumns) && (posy<numCRows)) {
		//numBRows or numAColumn would work, as they are equal
		for (k=0; k<numBRows; k++) { 
			sum += (A[posy*numAColumns+k] * B[k*numBColumns+posx]);
		}
	
		C[cpos] = sum;
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
	hostC = (float *) malloc(numCRows*numCColumns*sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
	wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);	//test

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	wbCheck(cudaMalloc((void**)&deviceA,sizeof(float)*numARows*numAColumns));
	wbCheck(cudaMalloc((void**)&deviceB,sizeof(float)*numBRows*numBColumns));
	wbCheck(cudaMalloc((void**)&deviceC,sizeof(float)*numCRows*numCColumns));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	wbCheck(cudaMemcpy(deviceA,hostA,sizeof(float)*numARows*numAColumns,cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpy(deviceB,hostB,sizeof(float)*numBRows*numBColumns,cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
	/*
	 *personal note: dimGrid ==> blocks in grid
	 *				 dimBlock ==> threads in block
	 *
	 *Some specs for reference:
	 *Maximum block dimensions: 1024 x 1024 x 64
	 *Maximum grid dimensions: 65535 x 65535 x 65535 
	 *Maximum amount of threads per block: 1024
	 *
	 *According to above specs, a block can hold a max of 1024 threads.
	 *However, there will be a limitation on amount of warps schedulable
	 *in a SM, so actually having 16x16 = 256 threads will result in 
	 *improved performance.
	 *
	 *I can tests on 16 vs 32 for width and found that a width of 16 had
	 *a small bit of improvement (by ~0.3ms) for compute time. It may
	 *be considered negligible, but the most probable reason for such a 
	 *small difference may be that vector addition is too simple of a 
	 *concept to have to worry about warp scheduling -- after all, threads
	 *do not have to sync with each other, which results in entire blocks
	 *hanging while waiting for that thread. 
	 *
	 *In conclusion, 16 may be a better choice for this GPU but in the future
	 *with bigger warp sizes or increase in max number threads an SM can hold,
	 *32 may result in faster performance.
	*/
    int width = 16;
	dim3 dimBlock(width,width,1);	
	dim3 dimGrid( (int)((numCColumns-1)/width + 1) ,
				  (int)((numCRows-1)/width + 1) ,
				  1);
	wbLog(TRACE, "Grid dimensions: ", (int)((numCColumns-1)/width + 1), " x ", (int)((numCRows-1)/width + 1));
	
	
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	matrixMultiply<<<dimGrid,dimBlock>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	wbCheck(cudaMemcpy(hostC,deviceC,sizeof(float)*numCRows*numCColumns,cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

