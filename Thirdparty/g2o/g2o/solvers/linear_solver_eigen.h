// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef G2O_LINEAR_SOLVER_EIGEN_H
#define G2O_LINEAR_SOLVER_EIGEN_H

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "../core/linear_solver.h"
#include "../core/batch_stats.h"
#include "../stuff/timeutil.h"

#include "../core/eigen_types.h"

#include <iostream>
#include <vector>

namespace g2o {

/**
 * \brief linear solver which uses the sparse Cholesky solver from Eigen
 *
 * Has no dependencies except Eigen. Hence, should compile almost everywhere
 * without to much issues. Performance should be similar to CSparse, I guess.
 */
template <typename MatrixType>
class LinearSolverEigen: public LinearSolver<MatrixType>
{
  public:
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseMatrix;
    typedef Eigen::Triplet<double> Triplet;
    typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PermutationMatrix;
    /**
     * \brief Sub-classing Eigen's SimplicialLDLT to perform ordering with a given ordering
     */
    class CholeskyDecomposition : public Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper>
    {
      public:
        CholeskyDecomposition() : Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper>() {}
        using Eigen::SimplicialLDLT< SparseMatrix, Eigen::Upper>::analyzePattern_preordered;

/*    class CholeskyDecomposition : public Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper>
    {
      public:
        CholeskyDecomposition() : Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper>() {}
        using Eigen::SimplicialLLT< SparseMatrix, Eigen::Upper>::analyzePattern_preordered;
*/
        void analyzePatternWithPermutation(SparseMatrix& a, const PermutationMatrix& permutation)
        {
          m_Pinv = permutation;
          m_P = permutation.inverse();
          int size = a.cols();
          SparseMatrix ap(size, size);
          ap.selfadjointView<Eigen::Upper>() = a.selfadjointView<UpLo>().twistedBy(m_P);
          analyzePattern_preordered(ap, true);
        }
    };

  public:
    LinearSolverEigen() :
      LinearSolver<MatrixType>(),
      _init(true), _blockOrdering(false), _writeDebug(false)
    {
    }

    virtual ~LinearSolverEigen()
    {
    }

    virtual bool init()
    {
      _init = true;
      return true;
    }

    bool preSolve(const SparseBlockMatrix<MatrixType>& A, double* t) 
    {
      if (_init)
        _sparseMatrix.resize(A.rows(), A.cols());
      fillSparseMatrix(A, !_init);
      if (_init) // compute the symbolic composition once
        computeSymbolicDecomposition(A);
      _init = false;

      *t=get_monotonic_time();
      _cholesky.factorize(_sparseMatrix);
      if (_cholesky.info() != Eigen::Success) { // the matrix is not positive definite
        if (_writeDebug) {
          std::cerr << "Cholesky failure, writing debug.txt (Hessian loadable by Octave)" << std::endl;
          A.writeOctave("debug.txt");
        }
        return false;
      }
    }

    bool solve(const SparseBlockMatrix<MatrixType>& A, double* x, double* b)
    {
      // NOTE(alexmillane): Put this in a function so I can use the presolve
      //                    steps elsewhere.
      // Factorize
      double t;
      if (!preSolve(A, &t)) {
        return false;
      }

      // Solving the system
      VectorXD::MapType xx(x, _sparseMatrix.cols());
      VectorXD::ConstMapType bb(b, _sparseMatrix.cols());
      xx = _cholesky.solve(bb);

      // Stats
      G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
      if (globalStats) {
        globalStats->timeNumericDecomposition = get_monotonic_time() - t;
        globalStats->choleskyNNZ = _cholesky.matrixL().nestedExpression().nonZeros() + _sparseMatrix.cols(); // the elements of D
      }

      return true;
    }

    bool solveInverse(const SparseBlockMatrix<MatrixType>& A, Eigen::MatrixXd* AInv)
    {
      // DEBUG (alexmillane)
      std::cout << "In alex's solve inverse function." << std::endl;

      // Factorize
      double t;
      if (!preSolve(A, &t)) {
        return false;
      }

      // Allocating the identity
      int size = A.cols();
      SparseMatrix I(size, size);
      I.setIdentity();

      // DEBUG(alexmillane):
      // block size for debug
      //constexpr size_t block_size = 12;

      // Just a sample before
      //SparseMatrix blockBefore = I.block(0, 0, block_size, block_size);
      //std::cout << "blockBefore: " << std::endl << blockBefore << std::endl;

      // Doing the solve (in place)
      // NOTE(alexmillane): In inverting the matrix, all sparsity is lost
      //SparseMatrix AInvDense(_cholesky.solve(I));
      *AInv = _cholesky.solve(I);

      // Just a sample before
      //SparseMatrix blockAfter = AInv.block(0, 0, block_size, block_size);
      //std::cout << "blockAfter: " << std::endl << blockAfter << std::endl;

      // DEBUG
      //std::cout << "AInv.rows(): " << *AInv.rows() << std::endl;
      //std::cout << "AInv.cols(): " << *AInv.cols() << std::endl;
      //std::cout << "AInv.nonZeros(): " << *AInv.nonZeros() << std::endl;


      return true;

    }

    // TODO(alexmillane): This mean that the above classes need the concept of an eigen sparse matrix.. Not good.
    bool getCholeskyFactor(
        const SparseBlockMatrix<MatrixType>& A,
        Eigen::SparseMatrix<double, Eigen::ColMajor>* factor_ptr,
        Eigen::PermutationMatrix<Eigen::Dynamic>* P_ptr) {
      // DEBUG (alexmillane)
      //std::cout << "Getting cholesky factor." << std::endl;

      // Factorize
      double t;
      if (!preSolve(A, &t)) {
        return false;
      }

      // Cholesky factor - Get L - (LDLT factorization)
      Eigen::SparseMatrix<double, Eigen::ColMajor> L(_cholesky.matrixL());

      // Cholesky factor - Get D - (LDLT factorization)
      Eigen::VectorXd D_diag = _cholesky.vectorD();
      Eigen::VectorXd D_diag_sqrt = D_diag.cwiseSqrt();
      Eigen::SparseMatrix<double, Eigen::ColMajor> D_sqrt(A.rows(), A.rows());
      std::vector<Eigen::Triplet<double>> triplets;
      for (size_t i = 0; i < D_diag_sqrt.size(); i++) {
        triplets.emplace_back(Eigen::Triplet<double>(i, i, D_diag_sqrt[i]));
      }
      D_sqrt.setFromTriplets(triplets.begin(), triplets.end());

      // Factor Output
      *factor_ptr = L * D_sqrt;

      // Cholesky factor - Get P - (LDLT factorization)
      *P_ptr = _cholesky.permutationP();

      // DEBUG
      //std::cout << "D_diag: " << std::endl << D_diag << std::endl;
      //std::cout << "D_diag_sqrt: " << std::endl << D_diag_sqrt << std::endl;
      //std::cout << "D_sqrt: " << std::endl << D_sqrt << std::endl;
      //std::cout << "P_indices: " << std::endl << P_ptr->indices() << std::endl;

/*      // Testing the pointer to underlying data
      int* int_ptr = P_ptr->indices().data();
      for (size_t i = 0; i < P_ptr->rows(); i++) {
        std::cout << "int_ptr[i]: " << int_ptr[i] << std::endl;
      }*/
    }

    //! do the AMD ordering on the blocks or on the scalar matrix
    bool blockOrdering() const { return _blockOrdering;}
    void setBlockOrdering(bool blockOrdering) { _blockOrdering = blockOrdering;}

    //! write a debug dump of the system matrix if it is not SPD in solve
    virtual bool writeDebug() const { return _writeDebug;}
    virtual void setWriteDebug(bool b) { _writeDebug = b;}

  protected:
    bool _init;
    bool _blockOrdering;
    bool _writeDebug;
    SparseMatrix _sparseMatrix;
    CholeskyDecomposition _cholesky;

    /**
     * compute the symbolic decompostion of the matrix only once.
     * Since A has the same pattern in all the iterations, we only
     * compute the fill-in reducing ordering once and re-use for all
     * the following iterations.
     */
    void computeSymbolicDecomposition(const SparseBlockMatrix<MatrixType>& A)
    {
      double t=get_monotonic_time();
      if (! _blockOrdering) {
        std::cout << "Computing ordering on the RAW matrix." << std::endl;
        _cholesky.analyzePattern(_sparseMatrix);
      } else {
        std::cout << "Computing ordering on the BLOCK matrix." << std::endl;
        // block ordering with the Eigen Interface
        // This is really ugly currently, as it calls internal functions from Eigen
        // and modifies the SparseMatrix class
        Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> blockP;
        {
          // prepare a block structure matrix for calling AMD
          std::vector<Triplet> triplets;
          for (size_t c = 0; c < A.blockCols().size(); ++c){
            const typename SparseBlockMatrix<MatrixType>::IntBlockMap& column = A.blockCols()[c];
            for (typename SparseBlockMatrix<MatrixType>::IntBlockMap::const_iterator it = column.begin(); it != column.end(); ++it) {
              const int& r = it->first;
              if (r > static_cast<int>(c)) // only upper triangle
                break;
              triplets.push_back(Triplet(r, c, 0.));
            }
          }

          // call the AMD ordering on the block matrix.
          // Relies on Eigen's internal stuff, probably bad idea
          SparseMatrix auxBlockMatrix(A.blockCols().size(), A.blockCols().size());
          auxBlockMatrix.setFromTriplets(triplets.begin(), triplets.end());
          typename CholeskyDecomposition::CholMatrixType C;
          C = auxBlockMatrix.selfadjointView<Eigen::Upper>();
          Eigen::internal::minimum_degree_ordering(C, blockP);
        }

        int rows = A.rows();
        assert(rows == A.cols() && "Matrix A is not square");

        // Adapt the block permutation to the scalar matrix
        PermutationMatrix scalarP;
        scalarP.resize(rows);
        int scalarIdx = 0;
        for (int i = 0; i < blockP.size(); ++i) {
          const int& p = blockP.indices()(i);
          int base  = A.colBaseOfBlock(p);
          int nCols = A.colsOfBlock(p);
          for (int j = 0; j < nCols; ++j)
            scalarP.indices()(scalarIdx++) = base++;
        }
        assert(scalarIdx == rows && "did not completely fill the permutation matrix");
        // analyze with the scalar permutation
        _cholesky.analyzePatternWithPermutation(_sparseMatrix, scalarP);

      }
      G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
      if (globalStats)
        globalStats->timeSymbolicDecomposition = get_monotonic_time() - t;
    }

    void fillSparseMatrix(const SparseBlockMatrix<MatrixType>& A, bool onlyValues)
    {
      if (onlyValues) {
        A.fillCCS(_sparseMatrix.valuePtr(), true);
      } else {

        // create from triplet structure
        std::vector<Triplet> triplets;
        triplets.reserve(A.nonZeros());
        for (size_t c = 0; c < A.blockCols().size(); ++c) {
          int colBaseOfBlock = A.colBaseOfBlock(c);
          const typename SparseBlockMatrix<MatrixType>::IntBlockMap& column = A.blockCols()[c];
          for (typename SparseBlockMatrix<MatrixType>::IntBlockMap::const_iterator it = column.begin(); it != column.end(); ++it) {
            int rowBaseOfBlock = A.rowBaseOfBlock(it->first);
            const MatrixType& m = *(it->second);
            for (int cc = 0; cc < m.cols(); ++cc) {
              int aux_c = colBaseOfBlock + cc;
              for (int rr = 0; rr < m.rows(); ++rr) {
                int aux_r = rowBaseOfBlock + rr;
                if (aux_r > aux_c)
                  break;
                triplets.push_back(Triplet(aux_r, aux_c, m(rr, cc)));
              }
            }
          }
        }
        _sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());

      }
    }
};

} // end namespace

#endif
