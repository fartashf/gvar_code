/// TODO: support double?

#include <torch/extension.h>
#include <THC/THC.h>
//#undef NDEBUG

#include <cstdio>
#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

// #include <mkl.h> // TODO use cblas.h instead?

// #include <thrust/host_vector.h>
// #include <thrust/device_ptr.h>


// #include <ATen/CUDAStream.h>

// THCState *state;

// cublasHandle_t getCurrentCUDABlasHandle() {
//     return THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
// }


namespace cusvd
{

template<int success = CUSOLVER_STATUS_SUCCESS, class T, class Status> // , class A = Status(*)(P), class D = Status(*)(T)>
std::unique_ptr<T, Status(*)(T*)> unique_allocate(Status(allocator)(T**),  Status(deleter)(T*))
{
    T* ptr;
    auto stat = allocator(&ptr);
    TORCH_CHECK(stat == success);
    return {ptr, deleter};
}

template <class T>
std::unique_ptr<T, decltype(&cudaFree)> unique_cuda_ptr(size_t len) {
    T* ptr;
    auto stat = cudaMalloc(&ptr, sizeof(T) * len);
    TORCH_CHECK(stat == cudaSuccess);
    return {ptr, cudaFree};
}

// solve U S V = svd(A)  a.k.a. syevj, where A (m, n), U (m, m), S (min(m, n)), V (n, n)
// see also https://docs.nvidia.com/cuda/cusolver/index.html#gesvdj-example1
std::tuple<at::Tensor, at::Tensor, at::Tensor>
svdj_forward(at::Tensor a, bool is_sort, double tol=1e-7, int max_sweeps=100)
{
    TORCH_CHECK(a.is_cuda(), "only cuda tensor is supported");
    TORCH_CHECK(a.dtype() == at::kFloat, "only float is supported");

    auto handle_ptr = unique_allocate(cusolverDnCreate, cusolverDnDestroy);
    const auto A = a.contiguous().clone().transpose(0, 1).contiguous().transpose(0, 1);
    const int econ = 0 ; /* econ = 1 for economy size */
    // const auto A = a;
    // const auto batch_size = A.size(0);
    const auto m = A.size(0);
    // TORCH_CHECK(m <= 32, "matrix row should be <= 32");
    const auto n = A.size(1);
    // TORCH_CHECK(n <= 32, "matrix col should be <= 32");
    const auto lda = m;
    const auto d_A = A.data_ptr<float>();
    const auto minmn = std::min(m, n);
    auto s = at::empty({minmn}, a.type());
    auto d_s = s.data_ptr<float>();
    auto U = at::empty({m, m}, a.type());
    const auto d_U = U.data_ptr<float>();
    const auto ldu = m;
    auto V = at::empty({n, n}, a.type());
    const auto d_V = V.data_ptr<float>();
    const auto ldv = n;

    auto params = unique_allocate(cusolverDnCreateGesvdjInfo, cusolverDnDestroyGesvdjInfo);
    auto status = cusolverDnXgesvdjSetTolerance(params.get(), tol);
    TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status);
    status = cusolverDnXgesvdjSetMaxSweeps(params.get(), max_sweeps);
    TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status);
    status = cusolverDnXgesvdjSetSortEig(params.get(), is_sort);
    TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status);

    auto jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
    int lwork;
    auto status_buffer = cusolverDnSgesvdj_bufferSize(
        handle_ptr.get(),
        jobz,
        econ,
        m,
        n,
        d_A,
        lda,
        d_s,
        d_U,
        ldu,
        d_V,
        ldv,
        &lwork,
        params.get());
    TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status_buffer);
    auto work_ptr = unique_cuda_ptr<float>(lwork);
    auto info_ptr = unique_cuda_ptr<int>(1);
    status = cusolverDnSgesvdj(
        handle_ptr.get(),
        jobz,
        econ,
        m,
        n,
        d_A,
        lda,
        d_s,
        d_U,
        ldu,
        d_V,
        ldv,
        work_ptr.get(),
        lwork,
        info_ptr.get(),
        params.get());
    TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status);

    std::vector<int> hinfo(1);
    auto status_memcpy = cudaMemcpy(hinfo.data(), info_ptr.get(), sizeof(int), cudaMemcpyDeviceToHost);
    TORCH_CHECK(cudaSuccess == status_memcpy);

    for(int i = 0 ; i < 1; ++i)
    {
        if ( 0 == hinfo[i] )
        {
            continue;
        }
        else if ( 0 > hinfo[i] )
        {
            printf("Error: %d-th parameter is wrong \n", -hinfo[i]);
            TORCH_CHECK(false);
        }
        else
        {
            printf("WARNING: matrix %d, info = %d : Jacobi method does not converge \n", i, hinfo[i] );
        }
    }

    // U = U.contiguous().transpose(1, 2).contiguous().transpose(1, 2);
    // s = s.contiguous().transpose(0, 1).contiguous().transpose(0, 1);
    // V = V.contiguous().transpose(1, 2).contiguous().transpose(1, 2);
    U = U.contiguous().transpose(0, 1).contiguous();
    s = s.contiguous();
    V = V.contiguous().transpose(0, 1).contiguous();

    return std::make_tuple(U, s, V);
}



} // namespace cusvd


// generate wrappers
// FIXME do not use legacy preprocessor macro
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("svdj_forward", &cusvd::svdj_forward,
          "cusolver based svd implementation");
}
