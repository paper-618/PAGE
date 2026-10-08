#ifndef _CSR_MV_
#define _CSR_MV_

#include <stdio.h>
#include <stdlib.h>

void csr_mv(int m, int n, int *csrRowPtr, int *csrColIdx, MAT_VAL_TYPE *csrVal, MAT_VAL_TYPE *x, MAT_VAL_TYPE *sy)
{
    #pragma omp parallel for
    for(int i = 0; i < m; i++)
    {
        MAT_VAL_TYPE sum = 0;
        for(int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++)
        {
            sum += csrVal[j] * x[csrColIdx[j]];
        }
        sy[i] = sum;
    }
}
void csrSpa_mv(int m, int n, int startRow, int *csrRowPtr, int *csrColIdx, MAT_VAL_TYPE *csrVal, MAT_VAL_TYPE *x, MAT_VAL_TYPE *sy)
{
    #pragma omp parallel for
    for(int i = 0; i < m; i++)
    {

        MAT_VAL_TYPE sum = 0;
        for(int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++)
        {
            sum += csrVal[j] * x[csrColIdx[j]];
        }
        sy[i + startRow] = sum;
    }
}

#endif
