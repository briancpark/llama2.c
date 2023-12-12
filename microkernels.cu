








// void matmul(half *xout, half *x, half *w, int n, int d)
// {
//     int serialElements = divUp(n, 32);
//     dim3 block_dim(32, 4);
//     int blocks = divUp(d, 4);
//     mat_vec_kernel<<<blocks, block_dim>>>(xout, x, w, n, d, serialElements);
// }


// __global__ void mat_vec_kernel(half *output, half *input, half *weight, int n, int d, int numSerialElements)
// {
//     int index = blockIdx.x * blockDim.y + threadIdx.y;
//     if (index >= d)
//         return;

//     float sum = 0;
//     for (int i = 0; i < numSerialElements; i++)
//     {
//         int j = i * 32 + threadIdx.x;
//         if (j < n)
//             sum += ((float)weight[index * n + j]) * ((float)input[j]);
//     }

//     using WarpReduce = cub::WarpReduce<float>;
//     __shared__ typename WarpReduce::TempStorage temp;
//     sum = WarpReduce(temp).Sum(sum);

//     if (threadIdx.x == 0)
//         output[index] = (half)sum;
// }