void rmsnorm(float* o, float* x, float* weight, int size);
void matmul_swiglu(float* xout0, float* xout1, float* x, float* w0, float* w1, int n, int d);
void matmul(float *xout, float *x, float *w, int n, int d);

void matmul_32000_768(float *xout, float *x, float *w);
void matmul_768_2048(float *xout, float *x, float *w);
void matmul_768_768(float *xout, float *x, float *w);
void matmul_2048_768(float *xout, float *x, float *w);
void matmul_32000_768_add(float* resid, float *xout, float *x, float *w);
void matmul_768_2048_add(float* resid, float *xout, float *x, float *w);
void matmul_768_768_add(float* resid, float *xout, float *x, float *w);
void matmul_2048_768_add(float* resid, float *xout, float *x, float *w);


// LLaMA 7B
void matmul_4096_4096(float *xout, float *x, float *w);
void matmul_11008_4096(float *xout, float *x, float *w);
void matmul_4096_11008(float *xout, float *x, float *w);
void matmul_32000_4096(float *xout, float *x, float *w);

// void softmax(float* x, int size);
void matmul_add(float* resid, float* xout, float* x, float* w, int n, int d);
