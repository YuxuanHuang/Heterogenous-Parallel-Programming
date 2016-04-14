// MP 1
#include	<wb.h>
#include	<math.h>
#define wbCheck(stmt)  do {                                                \
                        cudaError_t err = stmt;                            \
                        if (err != cudaSuccess) {                          \
                            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
                            return -1;                                     \
                        }                                                  \
                       } while(0)


__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	//derivation of i is broken up into posx and posy for easier reading.
	int posx = blockIdx.x*blockDim.x + threadIdx.x;
	int posy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = posy*blockDim.x*gridDim.x+posx;
	if ( i < len) 
		out[i] = in1[i] + in2[i];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

	wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	//using the predefined deviceInput1, deviceInput2, deviceOutput
	//size is needed for cudaMalloc and cudaMemcpy calls.
	int size = inputLength * sizeof(float);
	wbCheck(cudaMalloc((void**)&deviceInput1,size));
	wbCheck(cudaMalloc((void**)&deviceInput2,size));
	wbCheck(cudaMalloc((void**)&deviceOutput,size));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	//using template: cudaMemcpy(to,from,size,cudaMemcpyHostToDevice),
	//where size is inputLength
	wbCheck(cudaMemcpy(deviceInput1,hostInput1,size,cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpy(deviceInput2,hostInput2,size,cudaMemcpyHostToDevice));
	//of course, no need to cudaMemcpy deviceOutput (yet).
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
	dim3 dimGrid( (int)(sqrt((inputLength-1)/(pow(width,2)))+1) ,
				  (int)(sqrt((inputLength-1)/(pow(width,2)))+1) ,
				  1);

    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	vecAdd<<<dimGrid,dimBlock>>>(deviceInput1,deviceInput2,deviceOutput,inputLength);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
	
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	wbCheck(cudaMemcpy(hostOutput,deviceOutput,size,cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(deviceInput1);
	cudaFree(deviceInput2);
	cudaFree(deviceOutput);


    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

