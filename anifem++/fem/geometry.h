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

namespace Ani{
    ///@return determinant of input matrix
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
    };

    template<typename ScalarType = double>
    struct Tetra: public Tetras<DenseMatrix<ScalarType>>{
        using BaseT = Tetras<DenseMatrix<ScalarType>>;
        using MatType = DenseMatrix<ScalarType>;

        Tetra(const MatType& XY0, const MatType& XY1, const MatType& XY2, const MatType& XY3): 
            BaseT(XY0, XY1, XY2, XY3, 1) {}
        Tetra(ScalarType* XY0, ScalarType* XY1, ScalarType* XY2, ScalarType* XY3): 
            BaseT(XY0, XY1, XY2, XY3, 1) {} 

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
    };


    template<typename ScalarType = double>
    Tetras<DenseMatrix<ScalarType>> make_tetras(ScalarType* XY0, ScalarType* XY1, ScalarType* XY2, ScalarType* XY3, int count = 1) { return {XY0, XY1, XY2, XY3, count}; }
    template<typename ScalarType = double>
    Tetras<DenseMatrix<const ScalarType>> make_tetras(const ScalarType* XY0, const ScalarType* XY1, const ScalarType* XY2, const ScalarType* XY3, int count = 1) { return {XY0, XY1, XY2, XY3, count}; }

    
    /// @brief Solve the problem AX = B for A=A^T > 0
    /// @param A is square symmetric positive definite matrix of NxN size
    /// @param B is matrix of right-hand side of NxM size
    /// @param N is dimesion of the problem
    /// @param M is count of columns in B matrix 
    /// @param X is memory for solution, may coincide with B
    /// @param mem additional memory of size N*(N+1)/2 for work, may coinside with A
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
    /// @param mem additional memory of size N*(N+1)/2 for work, may coinside with A
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
    /// @param A is square matrix of NxN size
    /// @param B is matrix of right-hand side of NxM size
    /// @param N is dimesion of the problem
    /// @param M is count of columns in B matrix 
    /// @param X is memory for solution, may coincide with B
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
    /// @param A is matrix of NxN size
    /// @param Inv is memory for NxN matrix result, may coinside with A
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
};

#endif //CARNUM_GEOMETRY_H
