def main_codegen():
    main_code = f'''
    #include "include/TurboFFT.h"
    '''

    main_code += '''
template <typename DataType>
void test_turbofft( DataType* input_d, DataType* output_d, DataType* output_turbofft,
                    DataType* twiddle_d, DataType* checksum, std::vector<long long int> param, 
                    long long int bs, int thread_bs, int ntest){
    long long int N = (1 << param[0]), threadblock_bs, Ni, WorkerFFTSize;
    long long int logN = param[0];
    long long int shared_size[3], griddims[3], blockdims[3]; 
    DataType* inputs[3] = {input_d, output_d, output_d + N * bs};
    DataType* outputs[3] = {output_d, output_d + N * bs, output_d};
    int kernel_launch_times = param[1];
    float gflops, elapsed_time, mem_bandwidth;
    cudaEvent_t fft_begin, fft_end;
    
    cublasHandle_t handle;      

    int M = 16;
    dim3 gridDim1((N + 255) / 256, bs / M, 1);
    '''
    
    main_code += '''
    TurboFFT_Kernel_Entry<DataType> entry;
    for(int i = 0; i < kernel_launch_times; ++i){
        threadblock_bs = param[5 + i];
        Ni = (1 << param[2 + i]); 
        WorkerFFTSize = param[8 + i]; 
        shared_size[i] = Ni * threadblock_bs * sizeof(DataType);
        
        blockdims[i] = (Ni * threadblock_bs) / WorkerFFTSize;
        // griddims[i] = 128 * min((2048 / blockdims[i]), ((64 * 1024 + shared_size[i] - 1) / shared_size[i]));
        long long int shared_per_SM = 160 * 1024;
        shared_per_SM = 128 * 1024;
        griddims[i] = min(108 * min((2048 / blockdims[i]), (shared_per_SM / shared_size[i])), 
                ((N * bs) + (Ni * threadblock_bs) - 1) / (Ni * threadblock_bs));
        '''
    main_code += f'''
        griddims[i] = ((((N * bs) + (Ni * threadblock_bs) - 1) / (Ni * threadblock_bs))) / thread_bs;
    '''
    main_code += '''
        cudaFuncAttributes attr;
        if(cudaFuncSetAttribute(entry.turboFFTArr[logN][i], cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size[i]))
        printf("Set DynamicSharedMem failed\\n");
        if(cudaFuncSetAttribute(entry.turboFFTArr[logN][i], cudaFuncAttributePreferredSharedMemoryCarveout, (shared_per_SM * 100) / (164 * 1024)))
        printf("Set smemCarveout failed\\n");
        cudaError_t get_attr_res = cudaFuncGetAttributes (&attr, entry.turboFFTArr[logN][i] );
        if(get_attr_res != 0)
        printf("get_attr_res = %d\\n", get_attr_res);
    }
    
    cudaEventCreate(&fft_begin);
    cudaEventCreate(&fft_end);
    #pragma unroll
    for(int i = 0; i < kernel_launch_times; ++i){
        entry.turboFFTArr[logN][i]<<<griddims[i], blockdims[i], shared_size[i]>>>(inputs[i], outputs[i], twiddle_d, checksum, bs, thread_bs);
    }

    cudaEventRecord(fft_begin);
    #pragma unroll
    for (int j = 0; j < ntest; ++j){
    '''

    main_code += '''
        #pragma unroll
        for(int i = 0; i < kernel_launch_times; ++i){
            entry.turboFFTArr[logN][i]<<<griddims[i], blockdims[i], shared_size[i]>>>(inputs[i], outputs[i], twiddle_d, checksum, bs, thread_bs);
            cudaDeviceSynchronize();
        }
    '''
    main_code += '''
        cudaDeviceSynchronize();
    }
    cudaEventRecord(fft_end);
    cudaEventSynchronize(fft_begin);
    cudaEventSynchronize(fft_end);
    cudaEventElapsedTime(&elapsed_time, fft_begin, fft_end);
    elapsed_time = elapsed_time / ntest;
    gflops = 5 * N * log2f(N) * bs / elapsed_time * 1000 / 1000000000.f;
    mem_bandwidth = (float)(N * bs * sizeof(DataType) * 2) / (elapsed_time) * 1000.f / 1000000000.f;
    printf("turboFFT, %d, %d, %8.3f, %8.3f, %8.3f\\n",  (int)log2f(N),  (int)log2f(bs), elapsed_time, gflops, mem_bandwidth);
    
    checkCudaErrors(cudaMemcpy((void*)output_turbofft, (void*)outputs[kernel_launch_times - 1], N * bs * sizeof(DataType), cudaMemcpyDeviceToHost));
}


template <typename DataType>
void TurboFFT_main(long long logN, long long N, long long bs, bool if_profile,
                    bool if_verify, bool if_bench, long long bs_end, long long bs_gap,
                    int thread_bs, std::string param_file_path){


    DataType* input, *output_turbofft, *output_cufft;
    DataType* input_d, *output_d, *twiddle_d;
    int ntest = 10;

    std::vector<std::vector<long long int>> params;
    '''
    main_code += '''
    params = utils::load_parameters(param_file_path);

    DataType* checksum_d, *checksum_h;
    cudaMalloc((void**)&checksum_d, sizeof(DataType) * 16384 * 2);
    checksum_h = (DataType*)calloc(16384 * 2, sizeof(DataType));
    DataType* dest = checksum_h;
    for(int i = 2; i <= (1 << 13); i *= 2){
        utils::getDFTMatrixChecksum(dest, i);
        dest += i;
    }
    // utils::printData<DataType>(checksum_h + 62, 64);
    cudaMemcpy((void*)checksum_d, (void*)checksum_h, sizeof(DataType) * 16384 * 2, cudaMemcpyHostToDevice);


    
    if(!if_bench){
        // Verification
        utils::initializeData<DataType>(input, input_d, output_d, output_turbofft, output_cufft, twiddle_d, N, bs_end);

        if(if_verify){
            test_turbofft<DataType>(input_d, output_d, output_turbofft, twiddle_d, checksum_d, params[logN], bs, thread_bs, 1);
            profiler::cufft::test_cufft<DataType>(input_d, output_d, output_cufft, N, bs, 1);            
            utils::compareData<DataType>(output_turbofft, output_cufft, N * bs, 1e-4);
        }
        // Profiling
        if(if_profile){
            long long int bs_begin = bs;
            for(bs = bs_begin; bs <= bs_end; bs += bs_gap)
            profiler::cufft::test_cufft<DataType>(input_d, output_d, output_cufft, N, bs, ntest);
            
            for(bs = bs_begin; bs <= bs_end; bs += bs_gap)
            test_turbofft<DataType>(input_d, output_d, output_turbofft, twiddle_d, checksum_d, params[logN], bs, thread_bs, ntest);
        }
    }
    
    if(if_bench){
        utils::initializeData<DataType>(input, input_d, output_d, output_turbofft, output_cufft, twiddle_d, 1 << 25, 16 + 3);
        N = 1;
        for(logN = 1; logN <= 25; ++logN){
            N *= 2;
            bs = 1;
            // bs = bs << (28-logN);
            for(int i = 0; i < 29 - logN; i += 1){
                // profiler::cufft::test_cufft<DataType>(input_d, output_d, output_cufft, N, bs, ntest);
                test_turbofft<DataType>(input_d, output_d, output_turbofft, twiddle_d, checksum_d, params[logN], bs, thread_bs, ntest);
                bs *= 2;
                // break; 
            }
        }

   

    }
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(twiddle_d);
    free(input);
    free(output_cufft);
    free(output_turbofft);
}

int main(int argc, char *argv[]){
    if (argc < 3) {
        std::cerr << "Usage: program_name N bs" << std::endl;
        return 1;
    }
    
    long long logN = std::atol(argv[1]); // Convert first argument to integer
    long long N = 1 << logN; // Convert first argument to integer
    long long bs = std::atol(argv[2]); // Convert second argument to integer
    bool if_profile = 1;
    bool if_verify = 0;
    bool if_bench = 0;
    long long bs_end = bs;
    long long bs_gap = 1;
    int datatype =  0;
    int thread_bs = 1;
    std::string param_file_path;
    
    if (argc >= 4) bs_end = std::atol(argv[3]);
    if (argc >= 5) bs_gap = std::atol(argv[4]);
    if (argc >= 6) if_profile = std::atol(argv[5]);
    if (argc >= 7) if_verify  = std::atol(argv[6]);
    if (argc >= 8) if_bench = std::atol(argv[7]);
    if (argc >= 9) datatype = std::atol(argv[8]);
    if (argc >= 10) thread_bs = std::atol(argv[9]);
'''
    main_code += '''
    if(datatype == 0) {
        param_file_path = "../include/param/A100/param_float2.csv";
        TurboFFT_main<float2>(logN, N, bs, if_profile, if_verify, if_bench, 
                            bs_end, bs_gap, thread_bs, param_file_path);
    }
    else {
        param_file_path = "../include/param/A100/param_double2.csv";
        TurboFFT_main<double2>(logN, N, bs, if_profile, if_verify, if_bench, 
                            bs_end, bs_gap, thread_bs, param_file_path);
    }
    
    return 0;
}


    '''
    return main_code

if __name__ == '__main__':
    main_code = main_codegen('float2')
    file_name = "../../../main.cu"
    with open(file_name, 'w') as f:
        f.write(main_code)
