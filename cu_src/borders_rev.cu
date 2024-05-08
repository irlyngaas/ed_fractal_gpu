
extern "C" __global__
void borders_rev(float *d_poss, const int numPoints, float *d_bor, const int imageX, const int imageY, const int padX, const int padY) {
    //gl_WorkGroupSize.x=blockDim.x = NTHREADS
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    //Num Blocks=1 Cuda Variable???
    uint stride = blockDim.x * 1;
    uint offset = 0;

    __shared__ float cacheMaxX[256];
    __shared__ float cacheMaxY[256];
    __shared__ float cacheMinX[256];
    __shared__ float cacheMinY[256];

    float maxX = *(d_poss+tid+0);
    float maxY = *(d_poss+tid+1);
    float minX = *(d_poss+tid+0);
    float minY = *(d_poss+tid+1);

    while(tid + offset < numPoints) {
        maxX = max(maxX, *(d_poss + tid + 0 + offset));
        maxY = max(maxY, *(d_poss + tid + 1 + offset));
        minX = min(minX, *(d_poss + tid + 0 + offset));
        minY = min(minY, *(d_poss + tid + 1 + offset));
        offset+= stride;

    }

    cacheMaxX[threadIdx.x] = maxX;
    cacheMaxY[threadIdx.x] = maxY;
    cacheMinX[threadIdx.x] = minX;
    cacheMinY[threadIdx.x] = minY;

    __syncthreads();
    
    uint i = blockIdx.x/2;

    while(i != 0) {
        cacheMaxX[threadIdx.x] = max(cacheMaxX[threadIdx.x], cacheMaxX[threadIdx.x + i]);
        cacheMaxY[threadIdx.x] = max(cacheMaxY[threadIdx.x], cacheMaxY[threadIdx.x + i]);
        cacheMinX[threadIdx.x] = min(cacheMinX[threadIdx.x], cacheMinX[threadIdx.x + i]);
        cacheMinY[threadIdx.x] = min(cacheMinY[threadIdx.x], cacheMinY[threadIdx.x + i]);
        __syncthreads();

    }

    if(threadIdx.x == 0) {
        *(d_bor + 0) = cacheMaxX[0];
        *(d_bor + 1) = cacheMaxY[0];
        *(d_bor + 2) = cacheMinX[0];
        *(d_bor + 3) = cacheMinY[0];
    }

    for(uint i = tid; i<numPoints; i+= stride){
        //xs / (xmax-xmin) * float(image_x-2*pad_x)+float(pad_x)
        *(d_poss+i) = *(d_poss+i) / (cacheMaxX[0] - cacheMinX[0]) * (imageX -2*padX) + padX;
        *(d_poss+i+1) = *(d_poss+i+1) / (cacheMaxY[0] - cacheMinY[0]) * (imageY -2*padY) + padY;

    }


}
