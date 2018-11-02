#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
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
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
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

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {   // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
 int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    #pragma omp parallel for
    for (int j = 0; j < 100; j++) {   // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
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

void Labwork::labwork2_GPU() {
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    printf("my number of of device :  %d\n", numDevices);
    for (int i = 0; i < numDevices; i++) {
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


__global__ void grayscale(uchar3 *input, uchar3 *output) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
   output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU() {
   //(1) 
   uchar3 * devInput;
   uchar3 * devGray;
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
   grayscale<<<dimGrid, dimBlock>>>(devInput, devGray);
   //(5)

   //(6)
   cudaMemcpy(outputImage, devGray, pixelCount * sizeof (uchar3), cudaMemcpyDeviceToHost);
   //(7)
   cudaFree(devInput);
   cudaFree(devGray);
}

__global__ void grayscale2D(uchar3 *input, uchar3 *output, int imageWidth, int imageHeight) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x + (imageWidth*(threadIdx.y+blockIdx.y*blockDim.y));
   if(threadIdx.x > imageHeight) return;
   if(threadIdx.y > imageWidth) return;
   output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
   output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork4_GPU() {
   uchar3 * devInput;
   uchar3 * devGray;
   int pixelCount = inputImage->width * inputImage->height;
   outputImage = static_cast<char *>(malloc(pixelCount * 3));
   cudaMalloc(&devInput, pixelCount * sizeof (uchar3));
   cudaMalloc(&devGray, pixelCount * sizeof (uchar3));
   cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof (uchar3), cudaMemcpyHostToDevice);
   dim3 dimBlock = dim3(32,32);
   dim3 dimGrid = dim3(inputImage->width/32+1, inputImage->height/32+1);
   grayscale2D<<<dimGrid, dimBlock>>>(devInput,
 devGray, inputImage->width, inputImage->height);
   cudaMemcpy(outputImage, devGray, pixelCount * sizeof (uchar3), cudaMemcpyDeviceToHost);
   cudaFree(devInput);
   cudaFree(devGray);
}

// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++) {
        for (int col = 0; col < inputImage->width; col++) {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = (inputImage->buffer[tid * 3] + inputImage->buffer[tid * 3 + 1] + inputImage->buffer[tid * 3 + 2])/3;
                    int coefficient = kernel[(y+3) * 7 + x + 3];
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

__global__ void gaussianBlur(uchar3 * input, uchar3 * output, int width, int height) {
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
                     0, 0, 1, 2, 1, 0, 0 };
   int sum = 0;
   int c = 0;
   for ( int x = -3 ; x < 3 ; x++) {
      for ( int y = -3 ; y< 3 ; y++ ) {
     int i = tidx + x;
           int j = tidy + y;
     if (x < 0) continue;
           if (x >= width) continue;
           if (y < 0) continue;
           if (y >= height) continue;
     int tid = j * width + i;
           unsigned char gray = (input[tid].x + input[tid].y + input[tid].x)/3;
           int coefficient = kernel[(y+3) * 7 + x + 3];
           sum = sum + gray * coefficient;
           c += coefficient;    
      }
   }
   sum /= c;
   output[posOut].x = output[posOut].y = output[posOut].z = sum;
}


void Labwork::labwork5_GPU() {
   uchar3 * devInput;
   uchar3 * devBlur;
   int pixelCount = inputImage->width * inputImage->height;
   outputImage = static_cast<char *>(malloc(pixelCount * 3));
   cudaMalloc(&devInput, pixelCount * sizeof (uchar3));
   cudaMalloc(&devBlur, pixelCount * sizeof (uchar3));
   cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof (uchar3), cudaMemcpyHostToDevice);
   
   dim3 dimBlock = dim3(32,32);
   dim3 dimGrid = dim3(inputImage->width/32+1, inputImage->height/32+1);
   gaussianBlur<<<dimGrid, dimBlock>>>(devInput, devBlur, inputImage->width, inputImage->height);
   cudaMemcpy(outputImage, devBlur, pixelCount * sizeof (uchar3), cudaMemcpyDeviceToHost);
   cudaFree(devInput);
   cudaFree(devBlur);
}


__global__ void gaussianBlurSharedMem(uchar3 * input, uchar3 * output, int width, int height, int kernel[]) {
   int tidx = threadIdx.x + blockIdx.x * blockDim.x;
   if (tidx >= width) return;
   int tidy = threadIdx.y + blockIdx.y * blockDim.y;
   if (tidy >= height) return;
   int posOut = tidx + tidy * width;
   __shared__ int skernel[49];
   if (posOut < 49) {
    skernel[posOut] = kernel[posOut];
   }
   __syncthreads();

   int sum = 0;
   int c = 0;
   for ( int x = -3 ; x < 3 ; x++) {
      for ( int y = -3 ; y< 3 ; y++ ) {
     int i = tidx + x;
           int j = tidy + y;
     if (x < 0) continue;
           if (x >= width) continue;
           if (y < 0) continue;
           if (y >= height) continue;
     int tid = j * width + i;
           unsigned char gray = (input[tid].x + input[tid].y + input[tid].x)/3;
           int coefficient = skernel[(y+3) * 7 + x + 3];
           sum = sum + gray * coefficient;
           c += coefficient;    
      }
   }
   sum /= c;
   output[posOut].x = output[posOut].y = output[posOut].z = sum;
}


void Labwork::labwork5_GPU_sharedMem() {
   uchar3 * devInput;
   uchar3 * devBlur;
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
                     0, 0, 1, 2, 1, 0, 0 };
   dim3 dimBlock = dim3(32,32);
   dim3 dimGrid = dim3(inputImage->width/32+1, inputImage->height/32+1);
   gaussianBlurSharedMem<<<dimGrid, dimBlock>>>(devInput, devBlur, inputImage->width, inputImage->height, kernel);
   cudaMemcpy(outputImage, devBlur, pixelCount * sizeof (uchar3), cudaMemcpyDeviceToHost);
   cudaFree(devInput);
   cudaFree(devBlur);
}

void Labwork::labwork6_GPU() {

}

void Labwork::labwork7_GPU() {

}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}