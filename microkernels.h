void matmul(float *xout, float *x, float *w, int n, int d);

void matmul_32000_768(float *xout, float *x, float *w);
void matmul_768_2048(float *xout, float *x, float *w);
void matmul_768_768(float *xout, float *x, float *w);
void matmul_2048_768(float *xout, float *x, float *w);

// LLaMA 7B
void matmul_4096_4096(float *xout, float *x, float *w);
void matmul_11008_4096(float *xout, float *x, float *w);
void matmul_4096_11008(float *xout, float *x, float *w);
void matmul_32000_4096(float *xout, float *x, float *w);