// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


__global__ void addArrays(float * out_array, float * scanned_sums, int len) {
	int t = threadIdx.x;
	int i = blockIdx.x*blockDim.x+t;
	
	__shared__ float temparray[BLOCK_SIZE];
	__shared__ float to_add; 
	
	if (blockIdx.x > 0)
		to_add = scanned_sums[blockIdx.x-1];
	else
		to_add = 0.0f;
		
	if (i < len)
		temparray[t] = out_array[i];
	else
		temparray[t] = 0.0f;
	
	__syncthreads();
	
	temparray[t] += to_add;
	
	if (i < len)
		out_array[i] = temparray[t];
}

__global__ void scan(float * input, float * output, float * blocksum, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
	
	__shared__ float shared_scan[2*BLOCK_SIZE];
	
	int t = threadIdx.x;
	int i = blockIdx.x*blockDim.x+t;
	if (i < len)
		shared_scan[t] = input[i];
	else
		shared_scan[t] = 0.0f;
	if (i+blockDim.x < len)
		shared_scan[blockDim.x+t] = input[i+blockDim.x];
	else
		shared_scan[blockDim.x+t] = 0.0f;
	
	__syncthreads();
	
	for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		int index = (t+1)*stride*2-1;
		if (index < 2*BLOCK_SIZE)
			shared_scan[index] += shared_scan[index-stride];
		__syncthreads();
	}
	
	for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
		int index = (t+1)*stride*2-1;
		if (index+stride < 2*BLOCK_SIZE) 
			shared_scan[index + stride] += shared_scan[index];
		__syncthreads();
	}
	
	__syncthreads();
	if (i < len)
		output[i] = shared_scan[t];
	//last element in block has the sum. store it in blocksum
	if (t == BLOCK_SIZE-1)
		blocksum[blockIdx.x] = shared_scan[t];
	
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
	int dimension = (numElements-1)/BLOCK_SIZE + 1;
	dim3 dimGrid( dimension, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
	
	wbLog(TRACE, "dimGrid: ", dimension);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
	//we will need an additional array for the sum of each block
	//as well as scanned sums.
	float * blk_sums;
	float * scan_sums;
	wbCheck(cudaMalloc((void**)&blk_sums, dimension*sizeof(float)));
	wbCheck(cudaMalloc((void**)&scan_sums, dimension*sizeof(float)));
	
	//the first kernel should compute total sum of all blocks.
	scan<<<dimGrid, dimBlock>>>(deviceInput,deviceOutput,blk_sums,numElements);
	cudaDeviceSynchronize();
	//second kernel does the scan on the sums. This time the 3rd parameter is
	//no longer needed so can throw anything in there.
	scan<<<dimGrid, dimBlock>>>(blk_sums,scan_sums,blk_sums,dimension);
    cudaDeviceSynchronize();
	//third kernel adds sum to the blocks.
	addArrays<<<dimGrid,dimBlock>>>(deviceOutput, scan_sums, numElements);
	cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");
	
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

