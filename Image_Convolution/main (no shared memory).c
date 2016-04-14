#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

//@@ INSERT CODE HERE
//28+4 = 32, which is a power of 2. The next highest power, 64, will result in
//a total of 0x4038 bytes, and is too big for shared memory, which has capacity of
//0x4000.
#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH+Mask_width-1)

__global__ void convolution_2D_kernel(float *P,float *N,
									  int height, int width, int pitch, int channels,
									  const float* __restrict__ M) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	//contrasting lecture, I prefer to use x and y for col and row, respectively.
	//At least for me, it is slower to think in terms of col and row.
	int y_o = blockIdx.y*O_TILE_WIDTH + ty;
	int x_o = blockIdx.x*O_TILE_WIDTH + tx;
	int y_i = y_o - 2;
	int x_i = x_o - 2;
	float outp = 0.0;
	
	//instantiating shared memory
	//__shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
	
	//3-channels, so need to iterate through each channel per tile block
	for (int ch=0;ch<channels;ch++) {
		outp = 0.0;
		
		/*  Try running without shared memory portion for now.
		for ( int t=0; t<(max(width,height)-1)/BLOCK_WIDTH+1; t++ ) {
			if ( (y_i >= 0 && y_i < height) &&
				(x_i >= 0 && x_i < width) ) {
				Ns[ty][tx] = N[(y_i*width + x_i)*channels+ch];
			}
			else
				Ns[ty][tx] = 0.0;
			__syncthreads();
			
			if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) 
				for (int i=0;i<Mask_width;i++)
					for (int j=0;i<Mask_width;j++)
						outp += M[i*Mask_width+j] * Ns[ty+i][tx+j];
		}
		__syncthreads();
		*/
		
		//This is the non-shared memory solution
		for (int y=0;y<Mask_width;y++) {
			for (int x=0;x<Mask_width;x++) {
				if ( (y_i+y >= 0 && y_i+y < height) && (x_i+x >= 0 && x_i+x < width) ) {
					outp += (N[ (((y_i+y)*width) + x_i+x)*channels + ch ] * M[ y*Mask_width+x ]);
				}
			}
		}
		
		//calculating boundary conditions for output
		if (y_o < height && x_o < width)
			P[ (y_o*width + x_o)*channels+ch ] = outp;	//min(max(outp,0.0),1.0)
		
	}
		
	/*
	//pseudocode conversion to program code (this is sequential)
	float outp = 0.0;
	int x_offset = y_offset = 0;
	for (int i=0;i<height;i++) {
		for (int j=0;j<width;j++) {
			for (int k=0;k<channels;k++) {
				outp = 0.0;
				for (int y=0-Mask_radius;y<Mask_radius;y++) {
					for (int x=0-Mask_radius;x<Mask_radius;x++) {
						x_offset = x+j;
						y_offset = y+i;
						
						if ( (x_offset >= 0 && x_offset < width) &&
							(y_offset >= 0 && y_offset < height) ) {
							outp += N[(y_offset*width+x_offset)*channels+k]*M[(Mask_radius+y)*Mask_width+x+Mask_radius];
						}
					}
				}
			//store output
			P[(i*width+j)*channels+k] = min(max(outp,0),1);
			}
		}
	}
	*/
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH);
	dim3 dimGrid((imageWidth-1)/O_TILE_WIDTH+1,
				 (imageHeight-1)/O_TILE_WIDTH+1,
				 1);
	
	int pitch = wbImage_getPitch(inputImage);
	wbLog(TRACE, "image dimensions (Width x Height): ",imageWidth,"x",imageHeight);
	wbLog(TRACE, "grid dim (width x height): ",(imageWidth-1)/O_TILE_WIDTH+1,"x",(imageHeight-1)/O_TILE_WIDTH+1);
	wbLog(TRACE, "input pitch: ",pitch);
	wbLog(TRACE, "output pitch: ",wbImage_getPitch(outputImage));
	wbLog(TRACE, "channels: ",imageChannels);
	wbLog(TRACE, "mask dimensions (width x height): ", maskColumns,"x",maskRows );
	convolution_2D_kernel<<<dimGrid,dimBlock>>>(deviceOutputImageData,deviceInputImageData,imageHeight,
												imageWidth,pitch,
												imageChannels,deviceMaskData);
	cudaThreadSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
