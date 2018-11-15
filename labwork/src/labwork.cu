#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv)
{
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2)
    {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 )
    {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum)
    {
    case 1:
        labwork.labwork1_CPU();
        labwork.saveOutputImage("labwork2-cpu-out.jpg");
        printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        timer.start();
        labwork.labwork1_OpenMP();
        labwork.saveOutputImage("labwork2-openmp-out.jpg");
        break;
    case 2:
        labwork.labwork2_GPU();
        break;
    case 3:
        labwork.labwork3_GPU();
        labwork.saveOutputImage("labwork3-gpu-out.jpg");
        break;
    case 4:
        labwork.labwork4_GPU();
        labwork.saveOutputImage("labwork4-gpu-out.jpg");
        break;
    case 5:
        labwork.labwork5_CPU();
        labwork.saveOutputImage("labwork5-cpu-out.jpg");
        printf("labwork 5 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        timer.start();
        labwork.labwork5_GPU();
        printf("labwork %d GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        timer.start();
        labwork.labwork5_GPU_sharedMem();
        printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork5-gpu-out.jpg");
        break;
    case 6:
        labwork.labwork6_GPU();
        labwork.saveOutputImage("labwork6-gpu-out.jpg");
        break;
    case 7:
        labwork.labwork7_GPU();
        labwork.saveOutputImage("labwork7-gpu-out.jpg");
        break;
    case 8:
        labwork.labwork8_GPU();
        labwork.saveOutputImage("labwork8-gpu-out.jpg");
        break;
    case 9:
        labwork.labwork9_GPU();
        labwork.saveOutputImage("labwork9-gpu-out.jpg");
        break;
    case 10:
        labwork.labwork10_GPU();
        labwork.saveOutputImage("labwork10-gpu-out.jpg");
        break;

    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName)
{
    inputImage = jpegLoader.load(inputFileName);
    inputImage2 = jpegLoader.load("../data/iliketrain.jpg");
}

void Labwork::saveOutputImage(std::string outputFileName)
{
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

/********************
 *
 *    Labwork 1
 *
 ********************/

void Labwork::labwork1_CPU()
{
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++)     // let's do it 100 times, otherwise it's too fast!
    {
        for (int i = 0; i < pixelCount; i++)
        {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP()
{
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    #pragma omp parallel for
    for (int j = 0; j < 100; j++)     // let's do it 100 times, otherwise it's too fast!
    {
        for (int i = 0; i < pixelCount; i++)
        {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major)
    {
    case 2: // Fermi
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if (devProp.minor == 1) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

/********************
 *
 *    Labwork 2
 *
 ********************/

void Labwork::labwork2_GPU()
{
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    printf("my number of of device :  %d\n", numDevices);
    for (int i = 0; i < numDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device #%d\n", i);
        printf("- Name : %s\n", prop.name);
        printf("- Core Info\n");
        printf("    - SPCores : %d\n", getSPcores(prop));
        printf("    - ClockRate : %d\n", prop.clockRate);
        printf("    - MultiProcessor : %d\n", prop.multiProcessorCount);
        printf("    - WarpSize : %d\n", prop.warpSize);
        printf("- Memory Info\n");
        printf("    - ClockRate : %d\n", prop.memoryClockRate);
        printf("    - BusWidth : %d\n", prop.memoryBusWidth);
    }
}

/********************
 *
 *    Labwork 3
 *
 ********************/


__global__ void grayscale(uchar3 *input, uchar3 *output)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU()
{
    //(1)
    uchar3 *devInput;
    uchar3 *devGray;
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    //(2)
    cudaMalloc(&devInput, pixelCount * sizeof (uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof (uchar3));
    //(3)
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof (uchar3), cudaMemcpyHostToDevice);
    //(4)
    int dimBlock = 1024;
    int dimGrid = pixelCount / dimBlock;
    grayscale <<< dimGrid, dimBlock>>>(devInput, devGray);
    //(5)

    //(6)
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof (uchar3), cudaMemcpyDeviceToHost);
    //(7)
    cudaFree(devInput);
    cudaFree(devGray);
}

/********************
 *
 *    Labwork 4
 *
 ********************/

__global__ void grayscale2D(uchar3 *input, uchar3 *output, int imageWidth, int imageHeight)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x + (imageWidth * (threadIdx.y + blockIdx.y * blockDim.y));
    if(threadIdx.x > imageHeight) return;
    if(threadIdx.y > imageWidth) return;
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork4_GPU()
{
    uchar3 *devInput;
    uchar3 *devGray;
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    cudaMalloc(&devInput, pixelCount * sizeof (uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof (uchar3));
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof (uchar3), cudaMemcpyHostToDevice);
    dim3 dimBlock = dim3(32, 32);
    dim3 dimGrid = dim3(inputImage->width / 32 + 1, inputImage->height / 32 + 1);
    grayscale2D <<< dimGrid, dimBlock>>>(devInput,
                                         devGray, inputImage->width, inputImage->height);
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof (uchar3), cudaMemcpyDeviceToHost);
    cudaFree(devInput);
    cudaFree(devGray);
}

/********************
 *
 *    Labwork 5
 *
 ********************/

// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU()
{
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,
                     0, 3, 13, 22, 13, 3, 0,
                     1, 13, 59, 97, 59, 13, 1,
                     2, 22, 97, 159, 97, 22, 2,
                     1, 13, 59, 97, 59, 13, 1,
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0
                   };
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char *) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++)
    {
        for (int col = 0; col < inputImage->width; col++)
        {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++)
            {
                for (int x = -3; x <= 3; x++)
                {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = (inputImage->buffer[tid * 3] + inputImage->buffer[tid * 3 + 1] + inputImage->buffer[tid * 3 + 2]) / 3;
                    int coefficient = kernel[(y + 3) * 7 + x + 3];
                    sum = sum + gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int posOut = row * inputImage->width + col;
            outputImage[posOut * 3] = outputImage[posOut * 3 + 1] = outputImage[posOut * 3 + 2] = sum;
        }
    }
}

__global__ void gaussianBlur(uchar3 *input, uchar3 *output, int width, int height)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy >= height) return;
    int posOut = tidx + tidy * width;
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,
                     0, 3, 13, 22, 13, 3, 0,
                     1, 13, 59, 97, 59, 13, 1,
                     2, 22, 97, 159, 97, 22, 2,
                     1, 13, 59, 97, 59, 13, 1,
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0
                   };
    int sum = 0;
    int c = 0;
    for ( int x = -3 ; x < 3 ; x++)
    {
        for ( int y = -3 ; y < 3 ; y++ )
        {
            int i = tidx + x;
            int j = tidy + y;
            if (x < 0) continue;
            if (x >= width) continue;
            if (y < 0) continue;
            if (y >= height) continue;
            int tid = j * width + i;
            unsigned char gray = (input[tid].x + input[tid].y + input[tid].x) / 3;
            int coefficient = kernel[(y + 3) * 7 + x + 3];
            sum = sum + gray * coefficient;
            c += coefficient;
        }
    }
    sum /= c;
    output[posOut].x = output[posOut].y = output[posOut].z = sum;
}


void Labwork::labwork5_GPU()
{
    uchar3 *devInput;
    uchar3 *devBlur;
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    cudaMalloc(&devInput, pixelCount * sizeof (uchar3));
    cudaMalloc(&devBlur, pixelCount * sizeof (uchar3));
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof (uchar3), cudaMemcpyHostToDevice);

    dim3 dimBlock = dim3(32, 32);
    dim3 dimGrid = dim3(inputImage->width / 32 + 1, inputImage->height / 32 + 1);
    gaussianBlur <<< dimGrid, dimBlock>>>(devInput, devBlur, inputImage->width, inputImage->height);
    cudaMemcpy(outputImage, devBlur, pixelCount * sizeof (uchar3), cudaMemcpyDeviceToHost);
    cudaFree(devInput);
    cudaFree(devBlur);
}


__global__ void gaussianBlurSharedMem(uchar3 *input, uchar3 *output, int width, int height, int *kernel )
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy >= height) return;
    int posOut = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ int skernel[49];
    if (posOut < 49)
    {
        skernel[posOut] = kernel[posOut];
    }
    __syncthreads();

    int sum = 0;
    int c = 0;
    for ( int x = -3 ; x < 3 ; x++)
    {
        for ( int y = -3 ; y < 3 ; y++ )
        {
            int i = tidx + x;
            int j = tidy + y;
            if (x < 0) continue;
            if (x >= width) continue;
            if (y < 0) continue;
            if (y >= height) continue;
            int tid = j * width + i;
            unsigned char gray = (input[tid].x + input[tid].y + input[tid].x) / 3;
            int coefficient = skernel[(y + 3) * 7 + x + 3];
            sum = sum + gray * coefficient;
            c += coefficient;
        }
    }
    sum /= c;
    posOut = tidx + tidy * width;
    output[posOut].x = output[posOut].y = output[posOut].z = sum;
}


void Labwork::labwork5_GPU_sharedMem()
{
    uchar3 *devInput;
    uchar3 *devBlur;
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    cudaMalloc(&devInput, pixelCount * sizeof (uchar3));
    cudaMalloc(&devBlur, pixelCount * sizeof (uchar3));
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof (uchar3), cudaMemcpyHostToDevice);
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,
                     0, 3, 13, 22, 13, 3, 0,
                     1, 13, 59, 97, 59, 13, 1,
                     2, 22, 97, 159, 97, 22, 2,
                     1, 13, 59, 97, 59, 13, 1,
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0
                   };

    int *devKernel;
    cudaMalloc(&devKernel, sizeof(kernel));
    cudaMemcpy(devKernel, kernel, sizeof(kernel), cudaMemcpyHostToDevice);
    dim3 dimBlock = dim3(32, 32);
    dim3 dimGrid = dim3(inputImage->width / 32 + 1, inputImage->height / 32 + 1);
    gaussianBlurSharedMem <<< dimGrid, dimBlock>>>(devInput, devBlur, inputImage->width, inputImage->height, devKernel);
    cudaMemcpy(outputImage, devBlur, pixelCount * sizeof (uchar3), cudaMemcpyDeviceToHost);
    cudaFree(devInput);
    cudaFree(devBlur);
    cudaFree(devKernel);
}

/********************
 *
 *    Labwork 6
 *
 ********************/

__global__ void binari(uchar3 *input, uchar3 *output, int t)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int g = (int) ( input[tid].x / t ) * 255;
    output[tid].z = output[tid].y = output[tid].x = (char) g;
    //trash method
    /*if (input[tid].x < 127) {
    output[tid].x = output[tid].y = output[tid].y = 0;
    } else {
    output[tid].x = output[tid].y = output[tid].y = 255;
    }*/
}

__global__ void brightness(uchar3 *input, uchar3 *output, int brightnessCoef)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = output[tid].y = output[tid].z = min(max(input[tid].x + brightnessCoef, 0), 255);
    //trash method, again
    /*if (((int) input[tid].x + brightnessCoef) > 255) {
       output[tid].x = output[tid].y = output[tid].z = 255;
    } else if (((int) input[tid].x + brightnessCoef) < 0) {
       output[tid].x = output[tid].y = output[tid].z = 0;
    } else {
       output[tid].x = output[tid].y = output[tid].z = input[tid].x + brightnessCoef;
    }*/
}

__global__ void blending(uchar3 *input1, uchar3 *input2, uchar3 *output, double percent)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (percent * (double) input1[tid].x) + ((1.0 - percent) * (double) input2[tid].x);
    output[tid].y = (percent * (double) input1[tid].y) + ((1.0 - percent) * (double) input2[tid].y);
    output[tid].z = (percent * (double) input1[tid].z) + ((1.0 - percent) * (double) input2[tid].z);
}

void Labwork::labwork6_GPU()
{
    uchar3 *devInput;
    uchar3 *devGray;
    uchar3 *devBin;
    uchar3 *devBright;
    uchar3 *devInput2;
    uchar3 *devBlend;
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    cudaMalloc(&devInput, pixelCount * sizeof (uchar3));
    cudaMalloc(&devBin, pixelCount * sizeof (uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof (uchar3));
    cudaMalloc(&devBright, pixelCount * sizeof (uchar3));
    cudaMalloc(&devInput2, pixelCount * sizeof (uchar3));
    cudaMalloc(&devBlend, pixelCount * sizeof (uchar3));
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof (uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(devInput2, inputImage2->buffer, pixelCount * sizeof (uchar3), cudaMemcpyHostToDevice);
    int dimBlock = 1024;
    int dimGrid = pixelCount / dimBlock;
    //grayscale<<<dimGrid, dimBlock>>>(devInput, devGray);
    //binari<<<dimGrid, dimBlock>>>(devGray, devBin, 127);
    //brightness<<<dimGrid, dimBlock>>>(devGray, devBright, -10);
    blending <<< dimGrid, dimBlock>>>(devInput, devInput2, devBlend, 0.50);

    cudaMemcpy(outputImage, devBlend, pixelCount * sizeof (uchar3), cudaMemcpyDeviceToHost);
    cudaFree(devInput);
    cudaFree(devGray);
    cudaFree(devBin);
    cudaFree(devBright);
    cudaFree(devBlend);
    cudaFree(devInput2);

}

/********************
 *
 *    Labwork 7
 *
 ********************/

__global__ void uchar3ToInt(uchar3 * input, int * outputMin, int * outputMax, int width, int height) 
{
  // Get the global thread ID with half of block
  int tid = threadIdx.x + blockIdx.x * blockDim.x * 2; 
  if(tid+blockDim.x >= width * height) return;
  // Store the min/max from threadId of the current block and threadId of the "next" block
  outputMin[threadIdx.x + blockIdx.x * blockDim.x] = min(input[tid].x, input[tid + blockDim.x].x);
  outputMax[threadIdx.x + blockIdx.x * blockDim.x] = max(input[tid].x, input[tid + blockDim.x].x);
}

__global__ void reduceToMin(int * minTab) 
{
  extern __shared__ int cache[]; // Size of blockDim
  
  unsigned int localTid = threadIdx.x; // Local thread ID of the current block
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x * 2; // Global thread ID multiply by two because there is half of block
  
  cache[localTid] = min(minTab[tid], minTab[tid + blockDim.x]); // Store the min between the pair and the impair block
  
  __syncthreads(); // Synchronize to be sure cache[] is complete
  
  // REDUCE
  for (int r = blockDim.x  / 2; r > 0; r /= 2) 
  {
  
    if (localTid < r)
    {
      cache[localTid] = min(cache[localTid], cache[localTid + r]);
    }
    
    __syncthreads(); // Synchronize between each reduce step
  }
  
  // Store result
  if (localTid == 0)
  {
    minTab[blockIdx.x] = cache[0];
  }
}

__global__ void reduceToMax(int * maxTab) 
{
  extern __shared__ int cache[]; // Size of blockDim
  
  unsigned int localTid = threadIdx.x; // Local thread ID of the current block
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x * 2; // Global thread ID multiply by two because there is half of block
  cache[localTid] = max(maxTab[tid], maxTab[tid + blockDim.x]); // Store the max between the pair and the impair block
  
  __syncthreads(); // Synchronize to be sure cache[] is complete
  
  // REDUCE
  for (int r = blockDim.x  / 2; r > 0; r /= 2) 
  {
    if (localTid < r)
    {
      cache[localTid] = max(cache[localTid], cache[localTid + r]);
    }
    
    __syncthreads(); // Synchronize between each reduce step
  }
  
  // Store result
  if (localTid == 0)
  {
    maxTab[blockIdx.x] = cache[0];
  }
}

__global__ void grayscaleStretch(uchar3 * input, uchar3 * output, int * minG, int * maxG, int width, int height)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= width * height) return;
    unsigned char g = (double(input[tid].x - minG[0]) / double(maxG[0] - minG[0])) * 255;
    output[tid].x = output[tid].y = output[tid].z = g;
}

void Labwork::labwork7_GPU()
{
    // Host data
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    int dimBlock = 1024;
    int nbBlock = ceil((double)pixelCount / dimBlock);
    int cacheSize = dimBlock * sizeof(int);
    cudaError_t r = cudaGetLastError();
  
    // Device data
    uchar3 *devOutput;
    uchar3 *devImage;
    uchar3 *devGray;
    int *devMin;
    int *devMax;

    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));  
    cudaMalloc(&devImage, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
    cudaMalloc(&devMin, pixelCount * sizeof(int) / 2);
    cudaMalloc(&devMax, pixelCount * sizeof(int) / 2);  
  
    cudaMemcpy(devImage, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
  
    // Ask device to process data
    int currentNbBlock = nbBlock/2;
    // (1) Convert RGB to gray
    grayscale <<<nbBlock, dimBlock>>>(devImage, devGray);
    // (2) Extract min and max arrays by comparison with half of bock
    uchar3ToInt <<<currentNbBlock, dimBlock>>>(devGray, devMin, devMax, inputImage->width, inputImage->height);
    // (3) Apply REDUCE on min&max arrays until there are more than 1024 entries
    do
    {
      currentNbBlock /= 2;
      reduceToMin<<<currentNbBlock, dimBlock, cacheSize>>>(devMin);
      reduceToMax<<<currentNbBlock, dimBlock, cacheSize>>>(devMax);
    }while(currentNbBlock > 1);
    // (4) Stretch the gray image
    grayscaleStretch<<<nbBlock, dimBlock>>>(devGray, devOutput, devMin, devMax,  inputImage->width, inputImage->height);
  
    // Return data to host
    cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(devImage);
    cudaFree(devGray);
    cudaFree(devOutput);
    cudaFree(devMin);
    cudaFree(devMax);
    
    // Get the errors
    r = cudaGetLastError();
    if ( cudaSuccess != r ) {
      printf("ERROR : %s\n", cudaGetErrorString(r));
    }
}

/********************
 *
 *    Labwork 8
 *
 ********************/
 
struct hsv
{
  double *h;
  double *s;
  double *v;
};

__global__ void RGB2HSV(uchar3 *input, hsv output, int width, int height)
{
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int tidy = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = tidx + tidy * width;
    if(threadIdx.x > height || threadIdx.y > width) return;
    
    uchar3 rgb = input[tid];
    int maxRgb = max(rgb.x, max(rgb.y,rgb.z));
    
    double delta = (maxRgb - min(rgb.x, min(rgb.y,rgb.z))) / 255.0;
    double maxRgbReduced = maxRgb / 255.0;
    double R = rgb.x / 255.0;
    double G = rgb.y / 255.0;
    double B = rgb.z / 255.0;
    // Define the V
    output.v[tid] = maxRgbReduced;
    
    // Define the S
    if(maxRgb != 0)
    {

      output.s[tid] = delta / maxRgbReduced;
    } 
    else 
    {
      output.s[tid] = 0;
    }
    
    // Define the H
    if (delta == 0)
    {
      output.h[tid] = 0;
    }
    else if (maxRgb == rgb.x)
    {
      output.h[tid] = 60 * fmodf(((G - B) / delta), 6.0);
    }
    else if (maxRgb == rgb.y)
    {
      output.h[tid] = 60 * (((B - R) / delta) + 2);
    }
    else if (maxRgb == rgb.z)
    {
    output.h[tid] = 60 * (((R - G) / delta) + 4);
    }
    if (tid == 0)
    {
        printf("R : %d, G : %d, B : %d\n", input[tid].x, input[tid].y, input[tid].z);
        printf("maxRgbReduce : %lf, delta : %lf\n", maxRgbReduced, delta); 
      printf("H : %lf, S : %lf V : %lf\n", output.h[tid], output.s[tid], output.v[tid]);
    }
}

__global__ void HSV2RGB(hsv input, uchar3 *output, int width, int height)
{
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int tidy = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = tidx + tidy * width;
    if(threadIdx.x > height || threadIdx.y > width) return;
    
    double h = input.h[tid];
    double s = input.s[tid];
    double v = input.v[tid];
  
  double d = h / 60;
  int hi = (int) fmodf(d, 6);
  double f = d - hi;
  double l = v * (1 - s);
  double m = v * (1 - f * s);
  double n = v * (1 - (1 - f) * s);
  
  l = floor(l * 255 + 0.5); // Better approximation, 2.1 become 2 and 2.6 become 3
  m = floor(m * 255 + 0.5);
  n = floor(n * 255 + 0.5);
  int V = floor(v * 255 + 0.5); 
  
  if (h < 60)
  {
    output[tid].x = V;
    output[tid].y = n;
    output[tid].z = l;
  }
  else if (h < 120)
  {
    output[tid].x = m;
    output[tid].y = V;
    output[tid].z = l;
    return;
  }
  else if (h < 180)
  {
    output[tid].x = l;
    output[tid].y = V;
    output[tid].z = n;
  }
  else if (h < 240)
  {
    output[tid].x = l;
    output[tid].y = m;
    output[tid].z = V;
  }
  else if (h < 300)
  {
    output[tid].x = n;
    output[tid].y = l;
    output[tid].z = V;
  }
  else
  {
    output[tid].x = V;
    output[tid].y = l;
    output[tid].z = m;
  }
  
  if (tid == 0)
    {
      printf("H : %lf, S : %lf V : %lf\n", input.h[tid], input.s[tid], input.v[tid]);
        printf("d : %lf, f : %lf, l : %lf, m : %lf, n : %lf\n", d, f, l, m, n); 
      printf("R : %d, G : %d, B : %d\n", output[tid].x, output[tid].y, output[tid].z);
    }
}

void Labwork::labwork8_GPU()
{
  // Host data
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    dim3 dimBlock2d = dim3(32,32);
  dim3 nbBlock2d = dim3(ceil((double)inputImage->width/dimBlock2d.x), ceil((double)inputImage->height/dimBlock2d.y));
    cudaError_t r = cudaGetLastError();
  
    // Device data
    uchar3 *devRGB;
    uchar3 *devOutput;
    hsv devHSV;
    cudaMalloc(&devRGB, pixelCount * sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));  
    cudaMalloc((void**)&devHSV.h, pixelCount * sizeof(double));
    cudaMalloc((void**)&devHSV.s, pixelCount * sizeof(double));
    cudaMalloc((void**)&devHSV.v, pixelCount * sizeof(double));
    
    // Prepare data
    cudaMemcpy(devRGB, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

  // Process data
    RGB2HSV<<<nbBlock2d, dimBlock2d>>>(devRGB, devHSV, inputImage->width, inputImage->height);
    HSV2RGB<<<nbBlock2d, dimBlock2d>>>(devHSV, devOutput, inputImage->width, inputImage->height);

  // Get back the data
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);    
    
    // Free the device
    cudaFree(devRGB);
    cudaFree(devHSV.h);
    cudaFree(devHSV.s);
    cudaFree(devHSV.v);
    cudaFree(devOutput);
    
    // Show the error
    r = cudaGetLastError();
    if ( cudaSuccess != r ) {
      printf("ERROR : %s\n", cudaGetErrorString(r));
    }
}

/********************
 *
 *    Labwork 9
 *
 ********************/
 
struct histogram {
  int h[256];
};
 
__global__ void uchar3ToTabOfHisto(uchar3 *input, histogram *histo,int localHistoSize, int width, int height)
{
  int cache[256] = {0}; // Very slow :(
  
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; // Global thread ID multiply by two because there is half of block
  
  for (int i = 0; i < localHistoSize; i ++)
  {
    if (tid * localHistoSize + i >= width * height) continue;
    cache[input[tid * localHistoSize + i].x]++;
  }

  for (int i = 0; i < 256; i++)
  {
    histo[tid].h[i] = cache[i];
  }
}

__global__ void reduceHisto(histogram *histo, int nbHistoMax)
{
  unsigned int localTid = threadIdx.x;
  unsigned int tid = blockIdx.x;
  unsigned int halfOfNbHisto = ceil((double)nbHistoMax/2);
  if (tid + halfOfNbHisto >= nbHistoMax) return;
  histo[tid].h[localTid] += histo[tid + halfOfNbHisto].h[localTid];
}

__global__ void computeCDF(histogram *histo,int pixelCount)
{
  int minCdf = 0;
  int cumul = 0;
  
  for (int i = 0; i < 256; i++)
  {
    if (minCdf == 0)
    {
      minCdf = histo[0].h[i];
    }
    cumul += histo[0].h[i];
    histo[0].h[i] = round((double) (cumul - minCdf) / (pixelCount - minCdf) * 255.0);
  }
}

__global__ void equalizer(uchar3 *input, uchar3 *output, histogram *histo, int width, int height)
{ 
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= width * height) return;
  output[tid].x = output[tid].y = output[tid].z = histo[0].h[input[tid].x];
}

void Labwork::labwork9_GPU()
{
  // Host data
    int pixelCount = inputImage->width * inputImage->height;
  outputImage = static_cast<char *>(malloc(pixelCount * 3));
    int dimBlock = 1024;
    int localHistoSize = 1024;
    int nbBlock = ceil((double)pixelCount / dimBlock / localHistoSize); // local histo of one thread will considere 1024px
    cudaError_t r = cudaGetLastError();
    
    int currentDimBlock = 256;
    int currentNbHisto = ceil((double)pixelCount/localHistoSize);
    int currentNbBlock = currentNbHisto;
  
  // Device data
  uchar3 *devInput;
  uchar3 *devGray;
  uchar3 *devOutput;
  histogram *devHisto;
  cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
  cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
  cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
  cudaMalloc(&devHisto, currentNbHisto * sizeof(histogram));
  
  cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
  
  // Process data
  grayscale <<<nbBlock * 1024, dimBlock>>>(devInput, devGray);
    uchar3ToTabOfHisto <<<nbBlock, dimBlock>>>(devGray, devHisto,localHistoSize, inputImage->width, inputImage->height);
    
    do
    {
      currentNbHisto = currentNbBlock;
      currentNbBlock = ceil((double) currentNbBlock / 2);
      reduceHisto<<<currentNbBlock, currentDimBlock>>>(devHisto, currentNbHisto);
    }while(currentNbBlock > 1);
    computeCDF<<<1,1>>>(devHisto, pixelCount);
    equalizer<<<nbBlock * 1024, dimBlock>>>(devGray, devOutput, devHisto,  inputImage->width, inputImage->height);
  
    // Return data to host
    cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
  
  // Free the device
  cudaFree(devInput);
  cudaFree(devGray);
  cudaFree(devOutput);
  
  // Show the error
    r = cudaGetLastError();
    if ( cudaSuccess != r ) {
      printf("ERROR : %s\n", cudaGetErrorString(r));
    }
}

void Labwork::labwork10_GPU()
{

}