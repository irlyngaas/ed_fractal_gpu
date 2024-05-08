
extern "C" __global__
void compute_rev(const int PARAM_SIZE, const int numPoints, const float *map, const float *rnd, float *d_poss) {
    //gl_WorkGroupSize.x=blockDim.x = NTHREADS
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    //Num Blocks=1 Cuda Variable???
    uint stride = blockDim.x * 1;

    //float limit = 1.69999e+6;  

    //int mapp = mappings[0];
    __shared__ float s_map[5][7];

    
    for(int i=0; i<PARAM_SIZE; i++){
        for(int j=0; j<7; j++){
            s_map[i][j] = *(map+i*PARAM_SIZE+j);
            map++;
        }
    }
    __syncthreads();

    uint currentTarget = tid % PARAM_SIZE;
    float currentPoss[2];
    float newPoss[2];

    currentPoss[0] = s_map[currentTarget][4];
    currentPoss[1] = s_map[currentTarget][5];

    for(uint i = tid; i<numPoints; i+= stride){

        //d_color[i][0] = 0.5;
        //d_color[i][1] = 0.5;

        //FIX
        //float a = 12.9898*(rnd[0]);
        //float b = 78.233*(rnd[1]);
        //float c = 43758.5453;
        //float dt = currentPoss[0]*a + currentPoss[1]*b;
        //float sn = mod(dt,3.14);
        //float prob = floor(sin(sn) * c);
        //float prob = dt;
        float prob = rnd[i]; 
        float cump = 0.0;

        for(int j=0; j < PARAM_SIZE; j++) {
            cump += s_map[j][6];
            if(prob < cump) {
                currentTarget = j;
                break;
            }
        }

        newPoss[0] = s_map[currentTarget][0] * currentPoss[0] +
                     s_map[currentTarget][1] * currentPoss[1] +
                     s_map[currentTarget][4];
        newPoss[1] = s_map[currentTarget][2] * currentPoss[0] +
                     s_map[currentTarget][3] * currentPoss[1] +
                     s_map[currentTarget][4];

        //if(newPoss[0] > -limit && newPoss[0] < limit) {
        //    currentPoss[0] = newPoss[0];
        //}
        //else {
        //    newPoss[0] = 0.0;
        //    currentPoss[0] = newPoss[0];
        //}

        //if(newPoss[1] > -limit && newPoss[1] < limit) {
        //    currentPoss[1] = newPoss[1];
        //}
        //else {
        //    newPoss[1] = 0.0;
        //    currentPoss[1] = newPoss[1];
        //}

        *(d_poss+i) = newPoss[0];
        *(d_poss+i+1) = newPoss[1];

    }
}

extern "C" __global__
void compute_rev2(const int numPoints, const float *map, const float *rnd, float *d_poss) {
    //gl_WorkGroupSize.x=blockDim.x = NTHREADS
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    //Num Blocks=1 Cuda Variable???
    uint stride = blockDim.x * 1;

    float limit = 1.69999e+6;  

    //int mapp = mappings[0];
    __shared__ float s_map[2][7];

    
    for(int i=0; i<2; i++){
        for(int j=0; j<7; j++){
            s_map[i][j] = *(map+i*2+j);
            map++;
        }
    }
    __syncthreads();

    uint currentTarget = tid % 2;
    float currentPoss[2];
    float newPoss[2];

    currentPoss[0] = s_map[currentTarget][4];
    currentPoss[1] = s_map[currentTarget][5];

    for(uint i = tid; i<numPoints; i+= stride){

        //d_color[i][0] = 0.5;
        //d_color[i][1] = 0.5;

        //FIX
        float a = 12.9898*(rnd[0]);
        float b = 78.233*(rnd[1]);
        float c = 43758.5453;
        float dt = currentPoss[0]*a + currentPoss[1]*b;
        //float sn = mod(dt,3.14);
        //float prob = floor(sin(sn) * c);
        float prob = dt;
        float cump = 0.0;

        for(int j=0; j < 2; j++) {
            cump += s_map[j][6];
            if(prob < cump) {
                currentTarget = j;
                break;
            }
        }

        newPoss[0] = s_map[currentTarget][0] * currentPoss[0] +
                     s_map[currentTarget][1] * currentPoss[1] +
                     s_map[currentTarget][4];
        newPoss[1] = s_map[currentTarget][2] * currentPoss[0] +
                     s_map[currentTarget][3] * currentPoss[1] +
                     s_map[currentTarget][4];

        if(newPoss[0] > -limit && newPoss[0] < limit) {
            currentPoss[0] = newPoss[0];
        }
        else {
            newPoss[0] = 0.0;
            currentPoss[0] = newPoss[0];
        }

        if(newPoss[1] > -limit && newPoss[1] < limit) {
            currentPoss[1] = newPoss[1];
        }
        else {
            newPoss[1] = 0.0;
            currentPoss[1] = newPoss[1];
        }

        *(d_poss+i*numPoints) = newPoss[0];
        *(d_poss+i*numPoints+1) = newPoss[1];

    }
}

extern "C" __global__
void compute_rev4(const int numPoints, const float *map, const float *rnd, float *d_poss) {
    //gl_WorkGroupSize.x=blockDim.x = NTHREADS
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    //Num Blocks=1 Cuda Variable???
    uint stride = blockDim.x * 1;

    float limit = 1.69999e+6;  

    //int mapp = mappings[0];
    __shared__ float s_map[4][7];

    
    for(int i=0; i<4; i++){
        for(int j=0; j<7; j++){
            s_map[i][j] = *(map+i*4+j);
            map++;
        }
    }
    __syncthreads();

    uint currentTarget = tid % 4;
    float currentPoss[2];
    float newPoss[2];

    currentPoss[0] = s_map[currentTarget][4];
    currentPoss[1] = s_map[currentTarget][5];

    for(uint i = tid; i<numPoints; i+= stride){

        //d_color[i][0] = 0.5;
        //d_color[i][1] = 0.5;

        //FIX
        float a = 12.9898*(rnd[0]);
        float b = 78.233*(rnd[1]);
        float c = 43758.5453;
        float dt = currentPoss[0]*a + currentPoss[1]*b;
        //float sn = mod(dt,3.14);
        //float prob = floor(sin(sn) * c);
        float prob = dt;
        float cump = 0.0;

        for(int j=0; j < 4; j++) {
            cump += s_map[j][6];
            if(prob < cump) {
                currentTarget = j;
                break;
            }
        }

        newPoss[0] = s_map[currentTarget][0] * currentPoss[0] +
                     s_map[currentTarget][1] * currentPoss[1] +
                     s_map[currentTarget][4];
        newPoss[1] = s_map[currentTarget][2] * currentPoss[0] +
                     s_map[currentTarget][3] * currentPoss[1] +
                     s_map[currentTarget][4];

        if(newPoss[0] > -limit && newPoss[0] < limit) {
            currentPoss[0] = newPoss[0];
        }
        else {
            newPoss[0] = 0.0;
            currentPoss[0] = newPoss[0];
        }

        if(newPoss[1] > -limit && newPoss[1] < limit) {
            currentPoss[1] = newPoss[1];
        }
        else {
            newPoss[1] = 0.0;
            currentPoss[1] = newPoss[1];
        }

        *(d_poss+i*numPoints) = newPoss[0];
        *(d_poss+i*numPoints+1) = newPoss[1];

    }
}
extern "C" __global__
void compute_rev3(const int numPoints, const float *map, const float *rnd, float *d_poss) {
    //gl_WorkGroupSize.x=blockDim.x = NTHREADS
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    //Num Blocks=1 Cuda Variable???
    uint stride = blockDim.x * 1;

    float limit = 1.69999e+6;  

    //int mapp = mappings[0];
    __shared__ float s_map[3][7];

    
    for(int i=0; i<3; i++){
        for(int j=0; j<7; j++){
            s_map[i][j] = *(map+i*3+j);
            map++;
        }
    }
    __syncthreads();

    uint currentTarget = tid % 3;
    float currentPoss[2];
    float newPoss[2];

    currentPoss[0] = s_map[currentTarget][4];
    currentPoss[1] = s_map[currentTarget][5];

    for(uint i = tid; i<numPoints; i+= stride){

        //d_color[i][0] = 0.5;
        //d_color[i][1] = 0.5;

        //FIX
        float a = 12.9898*(rnd[0]);
        float b = 78.233*(rnd[1]);
        float c = 43758.5453;
        float dt = currentPoss[0]*a + currentPoss[1]*b;
        //float sn = mod(dt,3.14);
        //float prob = floor(sin(sn) * c);
        float prob = dt;
        float cump = 0.0;

        for(int j=0; j < 3; j++) {
            cump += s_map[j][6];
            if(prob < cump) {
                currentTarget = j;
                break;
            }
        }

        newPoss[0] = s_map[currentTarget][0] * currentPoss[0] +
                     s_map[currentTarget][1] * currentPoss[1] +
                     s_map[currentTarget][4];
        newPoss[1] = s_map[currentTarget][2] * currentPoss[0] +
                     s_map[currentTarget][3] * currentPoss[1] +
                     s_map[currentTarget][4];

        if(newPoss[0] > -limit && newPoss[0] < limit) {
            currentPoss[0] = newPoss[0];
        }
        else {
            newPoss[0] = 0.0;
            currentPoss[0] = newPoss[0];
        }

        if(newPoss[1] > -limit && newPoss[1] < limit) {
            currentPoss[1] = newPoss[1];
        }
        else {
            newPoss[1] = 0.0;
            currentPoss[1] = newPoss[1];
        }

        *(d_poss+i*numPoints) = newPoss[0];
        *(d_poss+i*numPoints+1) = newPoss[1];

    }
}

extern "C" __global__
void compute_rev5(const int numPoints, const float *map, const float *rnd, float *d_poss) {
    //gl_WorkGroupSize.x=blockDim.x = NTHREADS
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    //Num Blocks=1 Cuda Variable???
    uint stride = blockDim.x * 1;

    float limit = 1.69999e+6;  

    //int mapp = mappings[0];
    __shared__ float s_map[5][7];

    
    for(int i=0; i<5; i++){
        for(int j=0; j<7; j++){
            s_map[i][j] = *(map+i*5+j);
            map++;
        }
    }
    __syncthreads();

    uint currentTarget = tid % 5;
    float currentPoss[2];
    float newPoss[2];

    currentPoss[0] = s_map[currentTarget][4];
    currentPoss[1] = s_map[currentTarget][5];

    for(uint i = tid; i<numPoints; i+= stride){

        //d_color[i][0] = 0.5;
        //d_color[i][1] = 0.5;

        //FIX
        float a = 12.9898*(rnd[0]);
        float b = 78.233*(rnd[1]);
        float c = 43758.5453;
        float dt = currentPoss[0]*a + currentPoss[1]*b;
        //float sn = mod(dt,3.14);
        //float prob = floor(sin(sn) * c);
        float prob = dt;
        float cump = 0.0;

        for(int j=0; j < 5; j++) {
            cump += s_map[j][6];
            if(prob < cump) {
                currentTarget = j;
                break;
            }
        }

        newPoss[0] = s_map[currentTarget][0] * currentPoss[0] +
                     s_map[currentTarget][1] * currentPoss[1] +
                     s_map[currentTarget][4];
        newPoss[1] = s_map[currentTarget][2] * currentPoss[0] +
                     s_map[currentTarget][3] * currentPoss[1] +
                     s_map[currentTarget][4];

        if(newPoss[0] > -limit && newPoss[0] < limit) {
            currentPoss[0] = newPoss[0];
        }
        else {
            newPoss[0] = 0.0;
            currentPoss[0] = newPoss[0];
        }

        if(newPoss[1] > -limit && newPoss[1] < limit) {
            currentPoss[1] = newPoss[1];
        }
        else {
            newPoss[1] = 0.0;
            currentPoss[1] = newPoss[1];
        }

        *(d_poss+i*numPoints) = newPoss[0];
        *(d_poss+i*numPoints+1) = newPoss[1];

    }
}
