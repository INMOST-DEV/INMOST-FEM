//
// Created by Liogky Alexey on 08.04.2022.
//

#ifndef CARNUM_GEOMETRY_H
#define CARNUM_GEOMETRY_H

#ifdef WITH_EIGEN
#include <Eigen/Dense>
#endif

#include <algorithm> 
#include <stdexcept>
#include <cmath>
#include <limits>

#include "fem_memory.h"

#ifndef ANIFEM_JACOBI_MAX_SWEEPS
#define ANIFEM_JACOBI_MAX_SWEEPS 50
#endif

namespace Ani{
    /// @return determinant of input matrix
    template<typename Scalar>
    inline Scalar inverse3x3(const Scalar* m, Scalar* inv){
#define ID(I, J) (I) + 3*(J)
#define M(I, J) m[ID(I, J)]
        Scalar da = M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1);
        Scalar db = M(1, 2) * M(2, 0) - M(1, 0) * M(2, 2);
        Scalar dc = M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0);
        Scalar det = M(0, 0) * da + M(0, 1) * db + M(0, 2) * dc;

        inv[ID(0, 0)] = da / det;
        inv[ID(1, 0)] = db / det;
        inv[ID(2, 0)] = dc / det;

        for(int i = 1; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                inv[ID(j, i)] = (M((i + 1) % 3, (j + 1) % 3) * M((i + 2) % 3, (j + 2) % 3) -
                                M((i + 1) % 3, (j + 2) % 3) * M((i + 2) % 3, (j + 1) % 3)) / det;
#undef M
#undef ID
        return det;
    }

    /// Cross product axb = a x b
    template<typename Scalar>
    inline void cross(const Scalar* a, const Scalar* b, Scalar* axb){
        axb[0] = a[1]*b[2] - a[2]*b[1];
        axb[1] = -a[0]*b[2] + a[2]*b[0];
        axb[2] = a[0]*b[1] - a[1]*b[0];
    }

    /// Cross product axb = (p0 - p2) x (p1 - p2)
    template<typename Scalar>
    inline void cross(const Scalar* p0, const Scalar* p1, const Scalar* p2, Scalar* axb){
        Scalar  a[3]{p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2]},
                b[3]{p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]};
        axb[0] = a[1]*b[2] - a[2]*b[1];
        axb[1] = -a[0]*b[2] + a[2]*b[0];
        axb[2] = a[0]*b[1] - a[1]*b[0];
    }

    ///@return area of the triangle
    template<typename Scalar>
    inline Scalar tri_area(const Scalar* p0, const Scalar* p1, const Scalar* p2){
        Scalar  a[3]{p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2]},
                b[3]{p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]};
        Scalar c[3]{a[1]*b[2] - a[2]*b[1], -a[0]*b[2] + a[2]*b[0], a[0]*b[1] - a[1]*b[0]};
        return sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]) / 2;
    }

    ///@return area of the triangle defined by p0, p1, p2;
    /// set external normal on face defined by p0, p1, p2 in tetrahedron p0, p1, p2, p3
    template<typename Scalar>
    inline Scalar face_normal(const Scalar* p0, const Scalar* p1, const Scalar* p2, const Scalar* p3, Scalar* normal){
        Scalar  a[3]{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]},
                b[3]{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
        normal[0] = a[1]*b[2] - a[2]*b[1];
        normal[1] = -a[0]*b[2] + a[2]*b[0];
        normal[2] = a[0]*b[1] - a[1]*b[0];
        Scalar* n = normal;
        Scalar len =  sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        Scalar area = len / 2;
        for (int i = 0; i < 3; ++i) a[i] = p3[i] - p0[i];
        Scalar vol = n[0] * a[0] + n[1] * a[1] + n[2] * a[2];
        if (vol > 0)
            for (int i = 0; i < 3; ++i) normal[i] /= -len;
        else
            for (int i = 0; i < 3; ++i) normal[i] /= len;
        return area;
    }
    
    /// @brief Handler to store coordinates of nodes of collection of tetrahedrons
    template<   typename MatrTpXY0 = DenseMatrix<const double>, 
                typename MatrTpXY1 = DenseMatrix<const double>, 
                typename MatrTpXY2 = DenseMatrix<const double>, 
                typename MatrTpXY3 = DenseMatrix<const double>>
    struct TetrasCommon{
        const MatrTpXY0& XY0; 
        const MatrTpXY1& XY1;
        const MatrTpXY2& XY2;
        const MatrTpXY3& XY3;
        int fusion = 0;
        TetrasCommon(const MatrTpXY0& XY0, const MatrTpXY1& XY1, const MatrTpXY2& XY2, const MatrTpXY3& XY3, int fusion = 1): XY0{XY0}, XY1{XY1}, XY2{XY2}, XY3{XY3}, fusion(fusion) {}
    };
    template<typename ScalarType>
    struct TetrasCommon<DenseMatrix<ScalarType>, DenseMatrix<ScalarType>, DenseMatrix<ScalarType>, DenseMatrix<ScalarType>>{
        using MatrType = DenseMatrix<ScalarType>;
        MatrType XY0; 
        MatrType XY1;
        MatrType XY2;
        MatrType XY3;
        int fusion = 0;
        TetrasCommon(const MatrType& XY0, const MatrType& XY1, const MatrType& XY2, const MatrType& XY3): XY0{XY0}, XY1{XY1}, XY2{XY2}, XY3{XY3}, fusion(XY0.nCol) {
            assert(XY0.nRow == 3 && XY1.nRow == 3 && XY2.nRow == 3 && XY3.nRow == 3 && "Wrong coordinate arrays dimension, expected nRow = 3");
            assert(XY1.nCol == XY0.nCol && XY2.nCol == XY0.nCol && XY3.nCol == XY0.nCol && "Coordinate arrays should have same shape");
        }
        TetrasCommon(ScalarType* XY0, ScalarType* XY1, ScalarType* XY2, ScalarType* XY3, int count = 1): 
            XY0(XY0, 3, count), XY1(XY1, 3, count), XY2(XY2, 3, count), XY3(XY3, 3, count), fusion(count) {}
    };
    template<typename MatrTp = DenseMatrix<const double>>
    struct Tetras: public TetrasCommon<MatrTp, MatrTp, MatrTp, MatrTp> {
        using MatrType = MatrTp;
        using BaseT = TetrasCommon<MatrTp, MatrTp, MatrTp, MatrTp>;

        Tetras(const MatrTp& XY0, const MatrTp& XY1, const MatrTp& XY2, const MatrTp& XY3, int fusion = 1): 
            TetrasCommon<MatrTp, MatrTp, MatrTp, MatrTp>(XY0, XY1, XY2, XY3, fusion) {}
    };
    template<typename ScalarType>
    struct Tetras<DenseMatrix<ScalarType>>: public TetrasCommon<DenseMatrix<ScalarType>, DenseMatrix<ScalarType>, DenseMatrix<ScalarType>, DenseMatrix<ScalarType>> {
        using MatType = DenseMatrix<ScalarType>;
        using BaseT = TetrasCommon<MatType, MatType, MatType, MatType>;

        Tetras(const MatType& XY0, const MatType& XY1, const MatType& XY2, const MatType& XY3, int fusion = 1): 
            BaseT(XY0, XY1, XY2, XY3, fusion) {}
        Tetras(ScalarType* XY0, ScalarType* XY1, ScalarType* XY2, ScalarType* XY3, int count = 1): 
            BaseT(XY0, XY1, XY2, XY3, count) {}  
        Tetras(): Tetras(nullptr, nullptr, nullptr, nullptr, 0) {}    
    };

    template<typename ScalarType = double>
    struct Tetra: public Tetras<DenseMatrix<ScalarType>>{
        using BaseT = Tetras<DenseMatrix<ScalarType>>;
        using MatType = DenseMatrix<ScalarType>;

        Tetra(const MatType& XY0, const MatType& XY1, const MatType& XY2, const MatType& XY3): 
            BaseT(XY0, XY1, XY2, XY3, 1) {}
        Tetra(ScalarType* XY0, ScalarType* XY1, ScalarType* XY2, ScalarType* XY3): 
            BaseT(XY0, XY1, XY2, XY3, 1) {} 
        Tetra(): Tetra(nullptr, nullptr, nullptr, nullptr) {}   

        using NCScalar = typename std::remove_const<ScalarType>::type;   
        std::array<NCScalar, 3> centroid() const {
            std::array<NCScalar, 3> res;
            for (int k = 0; k < 3; ++k)
                res[k] = (BaseT::XY0[k] + BaseT::XY1[k] + BaseT::XY2[k] + BaseT::XY3[k]) / 4;
            return res;    
        }
        NCScalar diameter() const {
            auto len2 = [](const MatType& X0, const MatType& X1){
                NCScalar res = 0;
                for (int k = 0; k < 3; ++k)
                    res += (X1[k] - X0[k])*(X1[k] - X0[k]);
                return res;    
            };
            using B = BaseT;
            return sqrt(std::max({len2(B::XY0, B::XY1), len2(B::XY0, B::XY2), len2(B::XY0, B::XY3), 
                                  len2(B::XY1, B::XY2), len2(B::XY1, B::XY3), len2(B::XY2, B::XY3)}));
        } 
        /// @param iopposite_node is number (0, 1, 2 or 3) of opposite node relative to the face for which external normal is computing
        /// @warning returns non-normalized vector
        std::array<NCScalar, 3> normal_direction(unsigned char iopposite_node) const {
            ScalarType* X[4] = {BaseT::XY0.data, BaseT::XY1.data, BaseT::XY2.data, BaseT::XY3.data};
            auto i = iopposite_node;
            ScalarType* p0 = X[(i+1+0)%4], p1 = X[(i+1+1)%4], p2 = X[(i+1+2)%4], p3 = X[i%4];
            NCScalar    a[3]{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]},
                        b[3]{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]},
                        c[3]{p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]};
            std::array<NCScalar, 3> n{ a[1]*b[2] - a[2]*b[1], -a[0]*b[2] + a[2]*b[0], a[0]*b[1] - a[1]*b[0] };
            NCScalar vol = n[0] * c[0] + n[1] * c[1] + n[2] * c[2];
            if (vol > 0) for (int i = 0; i < 3; ++i) n[i] = -n[i];
            return n;
        }  
        /// @return normalized external normal to face formed by nodes (iopposite_node+1)%4, (iopposite_node+2)%4, (iopposite_node+3)%4
        std::array<NCScalar, 3> normal(unsigned char iopposite_node) const {
            ScalarType* X[4] = {BaseT::XY0.data, BaseT::XY1.data, BaseT::XY2.data, BaseT::XY3.data};
            std::array<NCScalar, 3> n;
            auto i = iopposite_node;
            face_normal(X[(i+1+0)%4], X[(i+1+1)%4], X[(i+1+2)%4], X[i%4], n.data());
            return n;
        }
    };


    template<typename ScalarType = double>
    Tetras<DenseMatrix<ScalarType>> make_tetras(ScalarType* XY0, ScalarType* XY1, ScalarType* XY2, ScalarType* XY3, int count = 1) { return {XY0, XY1, XY2, XY3, count}; }
    template<typename ScalarType = double>
    Tetras<DenseMatrix<const ScalarType>> make_tetras(const ScalarType* XY0, const ScalarType* XY1, const ScalarType* XY2, const ScalarType* XY3, int count = 1) { return {XY0, XY1, XY2, XY3, count}; }

    
    /// @brief Solve the problem AX = B for A=A^T > 0
    /// @param A is square symmetric positive definite matrix of NxN size
    /// @param B is col-major matrix of right-hand side of NxM size
    /// @param N is dimesion of the problem
    /// @param M is count of columns in B matrix 
    /// @param X is memory for col-major solution, may coincide with B
    /// @param mem additional memory of size N*(N+1)/2 for work, may coincide with A
    template<typename Scalar>
    inline void cholesky_solve(const Scalar* A, const Scalar* B, int N, int M, Scalar* X, Scalar* mem){
    #ifdef WITH_EIGEN
        (void) mem;
        using namespace Eigen;
        const Map<MatrixX<Scalar>> Am(const_cast<Scalar*>(A), N, N), Bm(const_cast<Scalar*>(B), N, M);
        Map<MatrixX<Scalar>> Xm(X, N, M);
        Xm = Am.llt().solve(Bm);
    #else
        Scalar* L = mem;
        #define ID(i, j) ( (i-j) + (2*N + 1 - j)*j/2 )
        for (uint k = 0; k < static_cast<uint>(N); ++k)
            for(uint i = k; i < static_cast<uint>(N); ++i)
                L[ID(i, k)] = A[i + k*N];       

        for (uint k = 0; k < static_cast<uint>(N); ++k){
            if (L[ID(k, k)] <= 0)
                throw std::runtime_error("Faced not positive definite matrix");
            L[ID(k, k)]  = sqrt(L[ID(k, k)]);
            for(uint i = k+1; i < static_cast<uint>(N); ++i)
				L[ID(i, k)] /= L[ID(k, k)];
            for(uint j = k+1; j < static_cast<uint>(N); ++j)
				for(uint i = j; i < static_cast<uint>(N); ++i)
					L[ID(i, j)] -= L[ID(i, k)]*L[ID(j, k)]; 
        }

        Scalar* Y = X;
        if (Y != B) 
            std::copy(B, B + N * M, Y);
        // LY=B
        for (uint i = 0; i < static_cast<uint>(N); ++i)
            for (uint k = 0; k < static_cast<uint>(M); ++k){
                for (uint j = 0; j < i; ++j)
                    Y[i + N*k] -= Y[j + N*k] * L[ID(i, j)];
                Y[i + k*N] /= L[ID(i, i)];    
            }    
        // L^TX = Y
        for (int i = N-1; i >= 0; --i)
            for (uint k = 0; k < static_cast<uint>(M); ++k){
                for (int j = N-1; j > i; --j)
                    X[i + k * N] -= X[j + k * N] * L[ID(j, i)];
                X[i + k*N] /= L[ID(i,i)];    
            }  
    #endif      
    }

    /// @brief Compute A^{-1} for matrix A=A^T > 0
    /// @param A is square symmetric positive definite matrix of NxN size
    /// @param Inv is memory for NxN matrix result
    /// @param N is dimension of problem
    /// @param mem additional memory of size N*(N+1)/2 for work, may coincide with A
    template<typename Scalar>
    inline void cholesky_inverse(const Scalar* A, Scalar* Inv, int N, Scalar* mem){
        std::fill(Inv, Inv + N*N, 0);
        for (uint i = 0; i < static_cast<uint>(N); ++i)
            Inv[i+i*N] = 1;
        cholesky_solve(A, Inv, N, N, Inv, mem);    
    }

    /// @brief Compute full pivote LU decomposition of a matrix 
    /// @param[in] A  is square matrix of NxN size
    /// @param[out] A is compact storage of (L\\U) decomposition 
    /// @param N is dimension of the matrix
    /// @param[in, out] p[N] is reorder of rows, on input should be defined, e.g. as p[i] = i
    /// @param[in, out] q[N] is reorder of cols, on input should be defined, e.g. as q[i] = i
    template<typename Scalar>
    inline void fullPivLU(Scalar* A, int N, int* p, int* q){
        #define A(i, j) A[(i) + (j)*N]
        for (int k = 0; k < N; k++){
            int pi = -1, pj = -1;
            Scalar max = 0.0;
            //find pivot in submatrix a(k:n,k:n)
            for (int i = k; i < N; i++)
                for (int j = k; j < N; j++)
                    if (fabs(A(i, j)) > max){
                        max = fabs(A(i, j));
                        pi = i;
                        pj = j;
                    }
            //Swap Row
            std::swap(p[k], p[pi]);
            for (int j = 0; j < N; j++)
                std::swap(A(k, j), A(pi, j));
            //Swap Col    
            std::swap(q[k], q[pj]);
            for (int i = 0; i < N; i++)
                std::swap(A(i, k), A(i, pj));
            Scalar max1 = 0.0;
            for (int i = k+1; i < N; i++) 
                max1 = std::max(max1, std::fabs(A(i, k)));    
            //END PIVOT
    
            //check pivot size and decompose
            if (max > 0.0 && max1 > 0.0 && max1 / max > std::numeric_limits<Scalar>::epsilon()){
                for (int i = k+1; i < N; i++){
                    //Column normalisation
                    Scalar tmp = ( A(i,k) /= A(k,k) ) ;
                    for (int j = k+1; j < N; j++){
                        //a(ik)*a(kj) subtracted from lower right submatrix elements
                        A(i, j) -= (tmp * A(k, j));
                    }
                }
            }
            //END DECOMPOSE
        }
        #undef A
    }

    /// @brief Compute solution for LU*X = B problem
    /// @param LU is compact storage of (L\\U) decomposition of matrix LU
    /// @param N is dimension of the matrix
    /// @param p[N] is reorder of rows
    /// @param q[N] is reorder of cols
    /// @param B is memory with matrix of right-hand side of NxM size, memory will be overriten
    /// @param X is memory for solution, shouldn't coincide with B
    template<typename Scalar>
    inline void LU_solve(const Scalar* LU, int N, const int* p, const int* q, Scalar* B, Scalar* X, int M){
        //x=Px
        for (int l = 0; l < M; ++l)
            for (int i = 0; i < N; ++i)
                X[i + N*l] = B[p[i] + N*l];
        //Lx=x
        for (int l = 0; l < M; ++l){
            int i = 0;
            for (; i < N && X[i + N*l] == 0.0; ++i) {};
            for (; i < N; ++i){
                Scalar v = X[i + N*l];
                for (int j = 0; j < i; ++j)
                    v -= LU[i + j*N]*X[j + N*l];
                X[i + N*l] = v;    
            }
        }
        //Ux=x
        for (int l = 0; l < M; ++l){
            X[(N-1) + N*l] /= LU[(N-1) + N*(N-1)];
            for (int i = N-2; i >= 0; --i){
                Scalar v = X[i + N*l];
                for (int j = i + 1; j < N; ++j)
                    v -= LU[i + N*j] * X[j + N*l];
                X[i + N*l] = v / LU[i + N*i];  
            }
        }
        //x=Qx
        std::copy(X, X + N*M, B);
        for (int l = 0; l < M; ++l)
            for (int i = 0; i < N; ++i)
                X[i + N*l] = B[q[i] + N*l];
    }

    /// @brief Solve the problem AX = B for sqaure dense matrix using LU decomposition
    /// @param A is square col-major matrix of NxN size
    /// @param B is col-major matrix of right-hand side of NxM size
    /// @param N is dimesion of the problem
    /// @param M is count of columns in B matrix 
    /// @param X is memory for col-major solution, may coincide with B
    /// @param mem additional real memory of size N*(N+M) for work
    /// @param imem additional integer memory of size 2*N for work
    template<typename Scalar>
    inline void fullPivLU_solve(const Scalar* A, const Scalar* B, int N, int M, Scalar* X, Scalar* mem, int* imem){
#ifdef WITH_EIGEN
        (void) mem; (void) imem;
        using namespace Eigen;
        const Map<MatrixX<Scalar>> Am(const_cast<Scalar*>(A), N, N), Bm(const_cast<Scalar*>(B), N, M);
        Map<MatrixX<Scalar>> Xm(X, N, M);
        Xm = Am.fullPivLu().solve(Bm);
#else        
        Scalar* Am = mem;
        int *p = imem, *q = imem + N;
        Scalar* Bm = mem + N*N;
        std::copy(A, A + N*N, Am);
        std::copy(B, B + N*M, Bm);
        for (int i = 0; i < N; ++i) p[i] = q[i] = i;
        fullPivLU(Am, N, p, q);
        LU_solve(Am, N, p, q, Bm, X, M);  
#endif
    }
    /// @brief Compute A^{-1} for dense matrix using LU decomposition
    /// @param A is col-major matrix of NxN size
    /// @param Inv is memory for NxN col-major matrix result, may coincide with A
    /// @param N is dimension of problem
    /// @param mem additional memory of size 2*N*N for work
    /// @param imem additional integer memory of size 2*N for work
    template<typename Scalar>
    inline void fullPivLU_inverse(const Scalar* A, Scalar* Inv, int N, Scalar* mem, int* imem){
#ifdef WITH_EIGEN
        (void) mem; (void) imem;
        using namespace Eigen;
        const Map<MatrixX<Scalar>> Am(const_cast<Scalar*>(A), N, N);
        Map<MatrixX<Scalar>> Invm(Inv, N, N);
        Invm = Am.fullPivLu().inverse();
#else         
        std::fill(Inv, Inv + N*N, 0);
        for (int i = 0; i < N; ++i)
            Inv[i+i*N] = 1;
        fullPivLU_solve(A, Inv, N, N, Inv, mem, imem);
#endif
    }

    /// @brief Compute Householder QR decomposition of dense matrix A = Q[1:N, 1:M] * R[1:M, 1:M]
    /// @param A is a col-major matrix of NxM size, N >= M
    /// @param N is number of rows in A
    /// @param M is number of cols in A
    /// @param Q is memory for NxN orthonormal col-major matrix result
    /// @param R is memory for MxM col-major matrix result
    /// @param mem additional real memory of size N*M + N + M for work
    template<typename Scalar>
    inline void householderQR(const Scalar* A, int N, int M, Scalar* Q, Scalar* R, Scalar* mem){
#ifdef WITH_EIGEN
        (void) mem;
        using namespace Eigen;
        const Map<const MatrixX<Scalar>> Am(A, N, M);
        HouseholderQR<MatrixX<Scalar>> qr(Am);
        Map<MatrixX<Scalar>> Qm(Q, N, N);
        Map<MatrixX<Scalar>> Rm(R, M, M);
        Qm = qr.householderQ();
        Rm = qr.matrixQR().topLeftCorner(M, M).template triangularView<Upper>();
#else
        #define W(i, j) mem[(i) + (j)*N]
        Scalar* tau = mem + N*M + N;
        Scalar* v = mem + N*M;

        std::copy(A, A + N*M, mem);

        for (int k = 0; k < M; ++k){
            Scalar norm = 0;
            for (int i = k; i < N; ++i)
                norm += W(i, k) * W(i, k);
            norm = std::sqrt(norm);
            if (norm <= std::numeric_limits<Scalar>::epsilon()){
                tau[k] = 0;
                v[k] = 1;
                for (int i = k + 1; i < N; ++i)
                    v[i] = 0;
                continue;
            }

            Scalar alpha = W(k, k);
            Scalar beta = (alpha >= 0) ? -norm : norm;
            tau[k] = (beta - alpha) / beta;
            Scalar inv_denom = Scalar(1) / (alpha - beta);
            v[k] = 1;
            for (int i = k + 1; i < N; ++i)
                v[i] = W(i, k) * inv_denom;

            W(k, k) = beta;
            for (int i = k + 1; i < N; ++i)
                W(i, k) = v[i];

            for (int j = k + 1; j < M; ++j){
                Scalar dot = v[k] * W(k, j);
                for (int i = k + 1; i < N; ++i)
                    dot += v[i] * W(i, j);
                dot *= tau[k];
                W(k, j) -= dot * v[k];
                for (int i = k + 1; i < N; ++i)
                    W(i, j) -= dot * v[i];
            }
        }

        for (int j = 0; j < M; ++j)
            for (int i = 0; i < M; ++i)
                R[i + j*M] = (i <= j) ? W(i, j) : 0;

        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                Q[i + j*N] = (i == j) ? 1 : 0;

        for (int k = M - 1; k >= 0; --k){
            if (tau[k] == 0)
                continue;
            v[k] = 1;
            for (int i = k + 1; i < N; ++i)
                v[i] = W(i, k);
            for (int j = 0; j < N; ++j){
                Scalar dot = v[k] * Q[k + j*N];
                for (int i = k + 1; i < N; ++i)
                    dot += v[i] * Q[i + j*N];
                dot *= tau[k];
                Q[k + j*N] -= dot * v[k];
                for (int i = k + 1; i < N; ++i)
                    Q[i + j*N] -= dot * v[i];
            }
        }
        #undef W
#endif
    }

#ifndef WITH_EIGEN
    namespace detail {
        template<typename Scalar>
        inline int jacobi_symmetric_eigen(Scalar* sym, int n, Scalar* eval, Scalar* evecs){
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    evecs[i + j*n] = (i == j) ? 1 : 0;

            const Scalar tol = Scalar(10) * n * n * std::numeric_limits<Scalar>::epsilon();
            int err = 1;
            for (int sweep = 0; sweep < ANIFEM_JACOBI_MAX_SWEEPS; ++sweep){
                Scalar off = 0;
                for (int p = 0; p < n; ++p)
                    for (int q = p + 1; q < n; ++q)
                        off += sym[p + q*n] * sym[p + q*n];
                if (off <= tol * tol){
                    err = 0;
                    break;
                }

                for (int p = 0; p < n; ++p){
                    for (int q = p + 1; q < n; ++q){
                        Scalar apq = sym[p + q*n];
                        if (std::fabs(apq) <= tol)
                            continue;
                        Scalar app = sym[p + p*n];
                        Scalar aqq = sym[q + q*n];
                        Scalar theta = Scalar(0.5) * std::atan2(Scalar(2) * apq, aqq - app);
                        Scalar c = std::cos(theta);
                        Scalar s = std::sin(theta);

                        for (int k = 0; k < n; ++k){
                            if (k == p || k == q)
                                continue;
                            Scalar gkp = sym[k + p*n];
                            Scalar gkq = sym[k + q*n];
                            sym[k + p*n] = sym[p + k*n] = c * gkp - s * gkq;
                            sym[k + q*n] = sym[q + k*n] = s * gkp + c * gkq;
                        }
                        sym[p + p*n] = c*c*app - Scalar(2)*s*c*apq + s*s*aqq;
                        sym[q + q*n] = s*s*app + Scalar(2)*s*c*apq + c*c*aqq;
                        sym[p + q*n] = sym[q + p*n] = 0;

                        for (int k = 0; k < n; ++k){
                            Scalar vkp = evecs[k + p*n];
                            Scalar vkq = evecs[k + q*n];
                            evecs[k + p*n] = c * vkp - s * vkq;
                            evecs[k + q*n] = s * vkp + c * vkq;
                        }
                    }
                }
            }

            if (err)
                return 1;

            for (int i = 0; i < n; ++i)
                eval[i] = sym[i + i*n];
            return 0;
        }

        template<typename Scalar>
        inline void sort_eigen_desc(Scalar* eval, Scalar* evecs, int n){
            for (int i = 0; i < n; ++i){
                int best = i;
                for (int j = i + 1; j < n; ++j)
                    if (eval[j] > eval[best])
                        best = j;
                if (best == i)
                    continue;
                std::swap(eval[i], eval[best]);
                for (int k = 0; k < n; ++k)
                    std::swap(evecs[k + i*n], evecs[k + best*n]);
            }
        }

        template<typename Scalar>
        inline void complete_orthonormal_cols(Scalar* q, int n, int n_done){
            const Scalar eps = std::numeric_limits<Scalar>::epsilon();
            for (int j = n_done; j < n; ++j){
                for (int i = 0; i < n; ++i)
                    q[i + j*n] = (i == (j - n_done) % n) ? 1 : 0;
                for (int c = 0; c < j; ++c){
                    Scalar dot = 0;
                    for (int i = 0; i < n; ++i)
                        dot += q[i + c*n] * q[i + j*n];
                    for (int i = 0; i < n; ++i)
                        q[i + j*n] -= dot * q[i + c*n];
                }
                Scalar norm = 0;
                for (int i = 0; i < n; ++i)
                    norm += q[i + j*n] * q[i + j*n];
                norm = std::sqrt(norm);
                if (norm <= eps){
                    for (int i = 0; i < n; ++i)
                        q[i + j*n] = (i == (j + 1) % n) ? 1 : 0;
                    for (int c = 0; c < j; ++c){
                        Scalar dot = 0;
                        for (int i = 0; i < n; ++i)
                            dot += q[i + c*n] * q[i + j*n];
                        for (int i = 0; i < n; ++i)
                            q[i + j*n] -= dot * q[i + c*n];
                    }
                    norm = 0;
                    for (int i = 0; i < n; ++i)
                        norm += q[i + j*n] * q[i + j*n];
                    norm = std::sqrt(norm);
                }
                if (norm > eps)
                    for (int i = 0; i < n; ++i)
                        q[i + j*n] /= norm;
            }
        }

        template<typename Scalar>
        inline void gram_ata(const Scalar* a, int n, int m, Scalar* g){
            for (int j = 0; j < m; ++j){
                for (int k = j; k < m; ++k){
                    Scalar s = 0;
                    for (int i = 0; i < n; ++i)
                        s += a[i + j*n] * a[i + k*n];
                    g[j + k*m] = g[k + j*m] = s;
                }
            }
        }

        template<typename Scalar>
        inline void gram_aat(const Scalar* a, int n, int m, Scalar* g){
            for (int j = 0; j < n; ++j){
                for (int k = j; k < n; ++k){
                    Scalar s = 0;
                    for (int i = 0; i < m; ++i)
                        s += a[j + i*n] * a[k + i*n];
                    g[j + k*n] = g[k + j*n] = s;
                }
            }
        }
    }
#endif

    /// @brief Compute SVD of dense col-major matrix A = U * diag(S) * V^T
    /// @param A col-major matrix of N x M size
    /// @param N number of rows
    /// @param M number of columns
    /// @param U optional memory for N x N orthogonal matrix (col-major), nullptr to skip
    /// @param S memory for min(N, M) singular values in decreasing order
    /// @param V optional memory for M x M orthogonal matrix (col-major), nullptr to skip
    /// @param mem additional real memory of size 2*min(N,M)^2 + min(N,M) for work
    /// @return 0 on success, 1 if Jacobi eigensolver failed to converge
    template<typename Scalar>
    inline int jacobiSVD(const Scalar* A, int N, int M, Scalar* U, Scalar* S, Scalar* V, Scalar* mem){
        const int K = std::min(N, M);
#ifdef WITH_EIGEN
        (void) mem;
        using namespace Eigen;
        const Map<const MatrixX<Scalar>> Am(A, N, M);
        int flags = 0;
        if (U)
            flags |= ComputeFullU;
        if (V)
            flags |= ComputeFullV;
        JacobiSVD<MatrixX<Scalar>> svd(Am, flags);
        for (int i = 0; i < K; ++i)
            S[i] = svd.singularValues()[i];
        if (U){
            Map<MatrixX<Scalar>> Um(U, N, N);
            Um = svd.matrixU();
        }
        if (V){
            Map<MatrixX<Scalar>> Vm(V, M, M);
            Vm = svd.matrixV();
        }
        return 0;
#else
        const Scalar eps = std::numeric_limits<Scalar>::epsilon();
        Scalar* gram = mem;
        Scalar* evecs_buf = mem + K*K;
        Scalar* eval = mem + 2*K*K;

        if (N >= M){
            detail::gram_ata(A, N, M, gram);
            Scalar* evecs = (V != nullptr) ? V : evecs_buf;
            if (detail::jacobi_symmetric_eigen(gram, M, eval, evecs))
                return 1;
            detail::sort_eigen_desc(eval, evecs, M);
            for (int i = 0; i < K; ++i)
                S[i] = std::sqrt(std::max(Scalar(0), eval[i]));

            if (U != nullptr){
                for (int j = 0; j < M; ++j){
                    if (S[j] > eps){
                        for (int i = 0; i < N; ++i){
                            Scalar sum = 0;
                            for (int t = 0; t < M; ++t)
                                sum += A[i + t*N] * evecs[t + j*M];
                            U[i + j*N] = sum / S[j];
                        }
                    } else {
                        for (int i = 0; i < N; ++i)
                            U[i + j*N] = 0;
                    }
                }
                detail::complete_orthonormal_cols(U, N, M);
            }
        } else {
            detail::gram_aat(A, N, M, gram);
            Scalar* evecs = (U != nullptr) ? U : evecs_buf;
            if (detail::jacobi_symmetric_eigen(gram, N, eval, evecs))
                return 1;
            detail::sort_eigen_desc(eval, evecs, N);
            for (int i = 0; i < K; ++i)
                S[i] = std::sqrt(std::max(Scalar(0), eval[i]));

            if (V != nullptr){
                for (int j = 0; j < N; ++j){
                    if (S[j] > eps){
                        for (int i = 0; i < M; ++i){
                            Scalar sum = 0;
                            for (int t = 0; t < N; ++t)
                                sum += A[t + i*N] * evecs[t + j*N];
                            V[i + j*M] = sum / S[j];
                        }
                    } else {
                        for (int i = 0; i < M; ++i)
                            V[i + j*M] = 0;
                    }
                }
                detail::complete_orthonormal_cols(V, M, N);
            }
        }
        return 0;
#endif
    }
};

#endif //CARNUM_GEOMETRY_H
