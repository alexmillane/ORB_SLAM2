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

#include "sparse_optimizer.h"
#include <Eigen/LU>
#include <fstream>
#include <iomanip>

#include "../stuff/timeutil.h"
#include "../stuff/macros.h"
#include "../stuff/misc.h"

//DEBUG(alexmillane)
#include "dense_matrix_io.h"
#include "marginal_covariance_cholesky.h"

namespace g2o {

using namespace std;
using namespace Eigen;

template <typename Traits>
BlockSolver<Traits>::BlockSolver(LinearSolverType* linearSolver) :
  BlockSolverBase(),
  _linearSolver(linearSolver)
{
  // workspace
  _Hpp=0;
  _Hll=0;
  _Hpl=0;
  _HplCCS = 0;
  _HschurTransposedCCS = 0;
  _Hschur=0;
  _DInvSchur=0;
  //_HschurInv=0;
  _coefficients=0;
  _bschur = 0;
  _xSize=0;
  _numPoses=0;
  _numLandmarks=0;
  _sizePoses=0;
  _sizeLandmarks=0;
  _doSchur=true;
  _verbose= false;
  _t_marginalize_cum = 0.0;
  _t_solve_cum = 0.0;
  _t_landmark_delta_cum = 0.0;
}

template <typename Traits>
void BlockSolver<Traits>::resize(int* blockPoseIndices, int numPoseBlocks, 
              int* blockLandmarkIndices, int numLandmarkBlocks,
              int s)
{
  deallocate();

  resizeVector(s);

  if (_doSchur) {
    // the following two are only used in schur
    assert(_sizePoses > 0 && "allocating with wrong size");
    _coefficients = new double [s];
    _bschur = new double[_sizePoses];
  }

  _Hpp=new PoseHessianType(blockPoseIndices, blockPoseIndices, numPoseBlocks, numPoseBlocks);
  if (_doSchur) {
    _Hschur=new PoseHessianType(blockPoseIndices, blockPoseIndices, numPoseBlocks, numPoseBlocks);
    _Hll=new LandmarkHessianType(blockLandmarkIndices, blockLandmarkIndices, numLandmarkBlocks, numLandmarkBlocks);
    _DInvSchur = new SparseBlockMatrixDiagonal<LandmarkMatrixType>(_Hll->colBlockIndices());
    _Hpl=new PoseLandmarkHessianType(blockPoseIndices, blockLandmarkIndices, numPoseBlocks, numLandmarkBlocks);
    _HplCCS = new SparseBlockMatrixCCS<PoseLandmarkMatrixType>(_Hpl->rowBlockIndices(), _Hpl->colBlockIndices());
    _HschurTransposedCCS = new SparseBlockMatrixCCS<PoseMatrixType>(_Hschur->colBlockIndices(), _Hschur->rowBlockIndices());
#ifdef G2O_OPENMP
    _coefficientsMutex.resize(numPoseBlocks);
#endif
  }
}

template <typename Traits>
void BlockSolver<Traits>::deallocate()
{
  if (_Hpp){
    delete _Hpp;
    _Hpp=0;
  }
  if (_Hll){
    delete _Hll;
    _Hll=0;
  }
  if (_Hpl){
    delete _Hpl;
    _Hpl = 0;
  }
  if (_Hschur){
    delete _Hschur;
    _Hschur=0;
  }
  if (_DInvSchur){
    delete _DInvSchur;
    _DInvSchur=0;
  }
  if (_coefficients) {
    delete[] _coefficients;
    _coefficients = 0;
  }
  if (_bschur) {
    delete[] _bschur;
    _bschur = 0;
  }
  if (_HplCCS) {
    delete _HplCCS;
    _HplCCS = 0;
  }
  if (_HschurTransposedCCS) {
    delete _HschurTransposedCCS;
    _HschurTransposedCCS = 0;
  }
}

template <typename Traits>
BlockSolver<Traits>::~BlockSolver()
{
  delete _linearSolver;
  deallocate();
}

template <typename Traits>
bool BlockSolver<Traits>::buildStructure(bool zeroBlocks)
{
  assert(_optimizer);

  size_t sparseDim = 0;
  _numPoses=0;
  _numLandmarks=0;
  _sizePoses=0;
  _sizeLandmarks=0;
  int* blockPoseIndices = new int[_optimizer->indexMapping().size()];
  int* blockLandmarkIndices = new int[_optimizer->indexMapping().size()];

  for (size_t i = 0; i < _optimizer->indexMapping().size(); ++i) {
    OptimizableGraph::Vertex* v = _optimizer->indexMapping()[i];
    int dim = v->dimension();
    if (! v->marginalized()){
      v->setColInHessian(_sizePoses);
      _sizePoses+=dim;
      blockPoseIndices[_numPoses]=_sizePoses;
      ++_numPoses;
    } else {
      v->setColInHessian(_sizeLandmarks);
      _sizeLandmarks+=dim;
      blockLandmarkIndices[_numLandmarks]=_sizeLandmarks;
      ++_numLandmarks;
    }
    sparseDim += dim;
  }
  resize(blockPoseIndices, _numPoses, blockLandmarkIndices, _numLandmarks, sparseDim);
  delete[] blockLandmarkIndices;
  delete[] blockPoseIndices;

  // allocate the diagonal on Hpp and Hll
  int poseIdx = 0;
  int landmarkIdx = 0;
  for (size_t i = 0; i < _optimizer->indexMapping().size(); ++i) {
    OptimizableGraph::Vertex* v = _optimizer->indexMapping()[i];
    if (! v->marginalized()){
      //assert(poseIdx == v->hessianIndex());
      PoseMatrixType* m = _Hpp->block(poseIdx, poseIdx, true);
      if (zeroBlocks)
        m->setZero();
      v->mapHessianMemory(m->data());
      ++poseIdx;
    } else {
      LandmarkMatrixType* m = _Hll->block(landmarkIdx, landmarkIdx, true);
      if (zeroBlocks)
        m->setZero();
      v->mapHessianMemory(m->data());
      ++landmarkIdx;
    }
  }
  assert(poseIdx == _numPoses && landmarkIdx == _numLandmarks);

  // temporary structures for building the pattern of the Schur complement
  SparseBlockMatrixHashMap<PoseMatrixType>* schurMatrixLookup = 0;
  if (_doSchur) {
    schurMatrixLookup = new SparseBlockMatrixHashMap<PoseMatrixType>(_Hschur->rowBlockIndices(), _Hschur->colBlockIndices());
    schurMatrixLookup->blockCols().resize(_Hschur->blockCols().size());
  }

  // here we assume that the landmark indices start after the pose ones
  // create the structure in Hpp, Hll and in Hpl
  for (SparseOptimizer::EdgeContainer::const_iterator it=_optimizer->activeEdges().begin(); it!=_optimizer->activeEdges().end(); ++it){
    OptimizableGraph::Edge* e = *it;

    for (size_t viIdx = 0; viIdx < e->vertices().size(); ++viIdx) {
      OptimizableGraph::Vertex* v1 = (OptimizableGraph::Vertex*) e->vertex(viIdx);
      int ind1 = v1->hessianIndex();
      if (ind1 == -1)
        continue;
      int indexV1Bak = ind1;
      for (size_t vjIdx = viIdx + 1; vjIdx < e->vertices().size(); ++vjIdx) {
        OptimizableGraph::Vertex* v2 = (OptimizableGraph::Vertex*) e->vertex(vjIdx);
        int ind2 = v2->hessianIndex();
        if (ind2 == -1)
          continue;
        ind1 = indexV1Bak;
        bool transposedBlock = ind1 > ind2;
        if (transposedBlock){ // make sure, we allocate the upper triangle block
          swap(ind1, ind2);
        }
        if (! v1->marginalized() && !v2->marginalized()){
          PoseMatrixType* m = _Hpp->block(ind1, ind2, true);
          if (zeroBlocks)
            m->setZero();
          e->mapHessianMemory(m->data(), viIdx, vjIdx, transposedBlock);
          if (_Hschur) {// assume this is only needed in case we solve with the schur complement
            schurMatrixLookup->addBlock(ind1, ind2);
          }
        } else if (v1->marginalized() && v2->marginalized()){
          // RAINER hmm.... should we ever reach this here????
          LandmarkMatrixType* m = _Hll->block(ind1-_numPoses, ind2-_numPoses, true);
          if (zeroBlocks)
            m->setZero();
          e->mapHessianMemory(m->data(), viIdx, vjIdx, false);
        } else { 
          if (v1->marginalized()){ 
            PoseLandmarkMatrixType* m = _Hpl->block(v2->hessianIndex(),v1->hessianIndex()-_numPoses, true);
            if (zeroBlocks)
              m->setZero();
            e->mapHessianMemory(m->data(), viIdx, vjIdx, true); // transpose the block before writing to it
          } else {
            PoseLandmarkMatrixType* m = _Hpl->block(v1->hessianIndex(),v2->hessianIndex()-_numPoses, true);
            if (zeroBlocks)
              m->setZero();
            e->mapHessianMemory(m->data(), viIdx, vjIdx, false); // directly the block
          }
        }
      }
    }
  }

  if (! _doSchur)
    return true;

  _DInvSchur->diagonal().resize(landmarkIdx);
  _Hpl->fillSparseBlockMatrixCCS(*_HplCCS);

  for (size_t i = 0; i < _optimizer->indexMapping().size(); ++i) {
    OptimizableGraph::Vertex* v = _optimizer->indexMapping()[i];
    if (v->marginalized()){
      const HyperGraph::EdgeSet& vedges=v->edges();
      for (HyperGraph::EdgeSet::const_iterator it1=vedges.begin(); it1!=vedges.end(); ++it1){
        for (size_t i=0; i<(*it1)->vertices().size(); ++i)
        {
          OptimizableGraph::Vertex* v1= (OptimizableGraph::Vertex*) (*it1)->vertex(i);
          if (v1->hessianIndex()==-1 || v1==v)
            continue;
          for  (HyperGraph::EdgeSet::const_iterator it2=vedges.begin(); it2!=vedges.end(); ++it2){
            for (size_t j=0; j<(*it2)->vertices().size(); ++j)
            {
              OptimizableGraph::Vertex* v2= (OptimizableGraph::Vertex*) (*it2)->vertex(j);
              if (v2->hessianIndex()==-1 || v2==v)
                continue;
              int i1=v1->hessianIndex();
              int i2=v2->hessianIndex();
              if (i1<=i2) {
                schurMatrixLookup->addBlock(i1, i2);
              }
            }
          }
        }
      }
    }
  }

  _Hschur->takePatternFromHash(*schurMatrixLookup);
  delete schurMatrixLookup;
  _Hschur->fillSparseBlockMatrixCCSTransposed(*_HschurTransposedCCS);

  return true;
}

template <typename Traits>
bool BlockSolver<Traits>::updateStructure(const std::vector<HyperGraph::Vertex*>& vset, const HyperGraph::EdgeSet& edges)
{
  for (std::vector<HyperGraph::Vertex*>::const_iterator vit = vset.begin(); vit != vset.end(); ++vit) {
    OptimizableGraph::Vertex* v = static_cast<OptimizableGraph::Vertex*>(*vit);
    int dim = v->dimension();
    if (! v->marginalized()){
      v->setColInHessian(_sizePoses);
      _sizePoses+=dim;
      _Hpp->rowBlockIndices().push_back(_sizePoses);
      _Hpp->colBlockIndices().push_back(_sizePoses);
      _Hpp->blockCols().push_back(typename SparseBlockMatrix<PoseMatrixType>::IntBlockMap());
      ++_numPoses;
      int ind = v->hessianIndex();
      PoseMatrixType* m = _Hpp->block(ind, ind, true);
      v->mapHessianMemory(m->data());
    } else {
      std::cerr << "updateStructure(): Schur not supported" << std::endl;
      abort();
    }
  }
  resizeVector(_sizePoses + _sizeLandmarks);

  for (HyperGraph::EdgeSet::const_iterator it = edges.begin(); it != edges.end(); ++it) {
    OptimizableGraph::Edge* e = static_cast<OptimizableGraph::Edge*>(*it);

    for (size_t viIdx = 0; viIdx < e->vertices().size(); ++viIdx) {
      OptimizableGraph::Vertex* v1 = (OptimizableGraph::Vertex*) e->vertex(viIdx);
      int ind1 = v1->hessianIndex();
      int indexV1Bak = ind1;
      if (ind1 == -1)
        continue;
      for (size_t vjIdx = viIdx + 1; vjIdx < e->vertices().size(); ++vjIdx) {
        OptimizableGraph::Vertex* v2 = (OptimizableGraph::Vertex*) e->vertex(vjIdx);
        int ind2 = v2->hessianIndex();
        if (ind2 == -1)
          continue;
        ind1 = indexV1Bak;
        bool transposedBlock = ind1 > ind2;
        if (transposedBlock) // make sure, we allocate the upper triangular block
          swap(ind1, ind2);

        if (! v1->marginalized() && !v2->marginalized()) {
          PoseMatrixType* m = _Hpp->block(ind1, ind2, true);
          e->mapHessianMemory(m->data(), viIdx, vjIdx, transposedBlock);
        } else { 
          std::cerr << __PRETTY_FUNCTION__ << ": not supported" << std::endl;
        }
      }
    }

  }

  return true;
}

template <typename Traits>
bool BlockSolver<Traits>::solve(){
  //cerr << __PRETTY_FUNCTION__ << endl;
  if (! _doSchur){
    double t=get_monotonic_time();
    bool ok = _linearSolver->solve(*_Hpp, _x, _b);
    G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
    if (globalStats) {
      globalStats->timeLinearSolver = get_monotonic_time() - t;
      globalStats->hessianDimension = globalStats->hessianPoseDimension = _Hpp->cols();
    }
    return ok;
  }

  // schur thing

  // backup the coefficient matrix
  double t=get_monotonic_time();

  // Updating the schur matrix _Hschur by marginalizing out the landmarks
  updateSchur();

  if (_verbose) {
    double marginalize_time = get_monotonic_time()-t;
    cerr << "Solve [marginalize] = " << marginalize_time << endl;
    _t_marginalize_cum += marginalize_time;
    cerr << "Cumulative [marginalize] = " << _t_marginalize_cum << endl;
  }

  // _bschur = _b for calling solver, and not touching _b
  memcpy(_bschur, _b, _sizePoses * sizeof(double));
  for (int i=0; i<_sizePoses; ++i){
    _bschur[i]-=_coefficients[i];
  }

  G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
  if (globalStats){
    globalStats->timeSchurComplement = get_monotonic_time() - t;
  }

  t=get_monotonic_time();
  bool solvedPoses = _linearSolver->solve(*_Hschur, _x, _bschur);
  if (globalStats) {
    globalStats->timeLinearSolver = get_monotonic_time() - t;
    globalStats->hessianPoseDimension = _Hpp->cols();
    globalStats->hessianLandmarkDimension = _Hll->cols();
    globalStats->hessianDimension = globalStats->hessianPoseDimension + globalStats->hessianLandmarkDimension;
  }
  if (_verbose) {
    double solve_time = get_monotonic_time()-t;
    cerr << "Solve [decompose and solve] = " <<  solve_time << endl;
    _t_solve_cum += solve_time;
    cerr << "Cumulative [decompose and solve] = " << _t_solve_cum << endl;
  }

  if (! solvedPoses)
    return false;

  // _x contains the solution for the poses, now applying it to the landmarks to get the new part of the
  // solution;
  double* xp = _x;
  double* cp = _coefficients;

  double* xl=_x+_sizePoses;
  double* cl=_coefficients + _sizePoses;
  double* bl=_b+_sizePoses;

  // cp = -xp
  for (int i=0; i<_sizePoses; ++i)
    cp[i]=-xp[i];

  // cl = bl
  memcpy(cl,bl,_sizeLandmarks*sizeof(double));

  // cl = bl - Bt * xp
  //Bt->multiply(cl, cp);
  _HplCCS->rightMultiply(cl, cp);

  // xl = Dinv * cl
  memset(xl,0, _sizeLandmarks*sizeof(double));
  _DInvSchur->multiply(xl,cl);
  //_DInvSchur->rightMultiply(xl,cl);
  if (_verbose) {
    double landmark_delta_time = get_monotonic_time()-t;
    cerr << "Solve [landmark delta] = " <<  landmark_delta_time << endl;
    _t_landmark_delta_cum += landmark_delta_time;
    cerr << "Cumulative [landmark delta] = " << _t_landmark_delta_cum << endl;
  }

  return true;
}

template <typename Traits>
void BlockSolver<Traits>::updateSchur()
{
  // _Hschur = _Hpp, but keeping the pattern of _Hschur
  _Hschur->clear();
  _Hpp->add(_Hschur);

  //_DInvSchur->clear();
  memset (_coefficients, 0, _sizePoses*sizeof(double));
# ifdef G2O_OPENMP
# pragma omp parallel for default (shared) schedule(dynamic, 10)
# endif
  for (int landmarkIndex = 0; landmarkIndex < static_cast<int>(_Hll->blockCols().size()); ++landmarkIndex) {
    const typename SparseBlockMatrix<LandmarkMatrixType>::IntBlockMap& marginalizeColumn = _Hll->blockCols()[landmarkIndex];
    assert(marginalizeColumn.size() == 1 && "more than one block in _Hll column");

    // calculate inverse block for the landmark
    const LandmarkMatrixType * D = marginalizeColumn.begin()->second;
    assert (D && D->rows()==D->cols() && "Error in landmark matrix");
    LandmarkMatrixType& Dinv = _DInvSchur->diagonal()[landmarkIndex];
    Dinv = D->inverse();

    LandmarkVectorType  db(D->rows());
    for (int j=0; j<D->rows(); ++j) {
      db[j]=_b[_Hll->rowBaseOfBlock(landmarkIndex) + _sizePoses + j];
    }
    db=Dinv*db;

    assert((size_t)landmarkIndex < _HplCCS->blockCols().size() && "Index out of bounds");
    const typename SparseBlockMatrixCCS<PoseLandmarkMatrixType>::SparseColumn& landmarkColumn = _HplCCS->blockCols()[landmarkIndex];

    for (typename SparseBlockMatrixCCS<PoseLandmarkMatrixType>::SparseColumn::const_iterator it_outer = landmarkColumn.begin();
        it_outer != landmarkColumn.end(); ++it_outer) {
      int i1 = it_outer->row;

      const PoseLandmarkMatrixType* Bi = it_outer->block;
      assert(Bi);

      PoseLandmarkMatrixType BDinv = (*Bi)*(Dinv);
      assert(_HplCCS->rowBaseOfBlock(i1) < _sizePoses && "Index out of bounds");
      typename PoseVectorType::MapType Bb(&_coefficients[_HplCCS->rowBaseOfBlock(i1)], Bi->rows());
#    ifdef G2O_OPENMP
      ScopedOpenMPMutex mutexLock(&_coefficientsMutex[i1]);
#    endif
      Bb.noalias() += (*Bi)*db;

      assert(i1 >= 0 && i1 < static_cast<int>(_HschurTransposedCCS->blockCols().size()) && "Index out of bounds");
      typename SparseBlockMatrixCCS<PoseMatrixType>::SparseColumn::iterator targetColumnIt = _HschurTransposedCCS->blockCols()[i1].begin();

      typename SparseBlockMatrixCCS<PoseLandmarkMatrixType>::RowBlock aux(i1, 0);
      typename SparseBlockMatrixCCS<PoseLandmarkMatrixType>::SparseColumn::const_iterator it_inner = lower_bound(landmarkColumn.begin(), landmarkColumn.end(), aux);
      for (; it_inner != landmarkColumn.end(); ++it_inner) {
        int i2 = it_inner->row;
        const PoseLandmarkMatrixType* Bj = it_inner->block;
        assert(Bj); 
        while (targetColumnIt->row < i2 /*&& targetColumnIt != _HschurTransposedCCS->blockCols()[i1].end()*/)
          ++targetColumnIt;
        assert(targetColumnIt != _HschurTransposedCCS->blockCols()[i1].end() && targetColumnIt->row == i2 && "invalid iterator, something wrong with the matrix structure");
        PoseMatrixType* Hi1i2 = targetColumnIt->block;//_Hschur->block(i1,i2);
        assert(Hi1i2);
        (*Hi1i2).noalias() -= BDinv*Bj->transpose();
      }
    }
  }  
}

template <typename Traits>
bool BlockSolver<Traits>::computeMarginals(SparseBlockMatrix<MatrixXd>& spinv, const std::vector<std::pair<int, int> >& blockIndices)
{
  double t = get_monotonic_time();
  bool ok = _linearSolver->solvePattern(spinv, blockIndices, *_Hpp);
  G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
  if (globalStats) {
    globalStats->timeMarginals = get_monotonic_time() - t;
  }
  return ok;
}

template <typename Traits>
bool BlockSolver<Traits>::buildSystem()
{
  // clear b vector
# ifdef G2O_OPENMP
# pragma omp parallel for default (shared) if (_optimizer->indexMapping().size() > 1000)
# endif
  for (int i = 0; i < static_cast<int>(_optimizer->indexMapping().size()); ++i) {
    OptimizableGraph::Vertex* v=_optimizer->indexMapping()[i];
    assert(v);
    v->clearQuadraticForm();
  }
  _Hpp->clear();
  if (_doSchur) {
    _Hll->clear();
    _Hpl->clear();
  }

  // resetting the terms for the pairwise constraints
  // built up the current system by storing the Hessian blocks in the edges and vertices
# ifndef G2O_OPENMP
  // no threading, we do not need to copy the workspace
  JacobianWorkspace& jacobianWorkspace = _optimizer->jacobianWorkspace();
# else
  // if running with threads need to produce copies of the workspace for each thread
  JacobianWorkspace jacobianWorkspace = _optimizer->jacobianWorkspace();
# pragma omp parallel for default (shared) firstprivate(jacobianWorkspace) if (_optimizer->activeEdges().size() > 100)
# endif
  for (int k = 0; k < static_cast<int>(_optimizer->activeEdges().size()); ++k) {
    OptimizableGraph::Edge* e = _optimizer->activeEdges()[k];
    e->linearizeOplus(jacobianWorkspace); // jacobian of the nodes' oplus (manifold)
    e->constructQuadraticForm();
#  ifndef NDEBUG
    for (size_t i = 0; i < e->vertices().size(); ++i) {
      const OptimizableGraph::Vertex* v = static_cast<const OptimizableGraph::Vertex*>(e->vertex(i));
      if (! v->fixed()) {
        bool hasANan = arrayHasNaN(jacobianWorkspace.workspaceForVertex(i), e->dimension() * v->dimension());
        if (hasANan) {
          cerr << "buildSystem(): NaN within Jacobian for edge " << e << " for vertex " << i << endl;
          break;
        }
      }
    }
#  endif
  }

  // flush the current system in a sparse block matrix
# ifdef G2O_OPENMP
# pragma omp parallel for default (shared) if (_optimizer->indexMapping().size() > 1000)
# endif
  for (int i = 0; i < static_cast<int>(_optimizer->indexMapping().size()); ++i) {
    OptimizableGraph::Vertex* v=_optimizer->indexMapping()[i];
    int iBase = v->colInHessian();
    if (v->marginalized())
      iBase+=_sizePoses;
    v->copyB(_b+iBase);
  }

  return 0;
}


template <typename Traits>
bool BlockSolver<Traits>::setLambda(double lambda, bool backup)
{
  if (backup) {
    _diagonalBackupPose.resize(_numPoses);
    _diagonalBackupLandmark.resize(_numLandmarks);
  }
# ifdef G2O_OPENMP
# pragma omp parallel for default (shared) if (_numPoses > 100)
# endif
  for (int i = 0; i < _numPoses; ++i) {
    PoseMatrixType *b=_Hpp->block(i,i);
    if (backup)
      _diagonalBackupPose[i] = b->diagonal();
    b->diagonal().array() += lambda;
  }
# ifdef G2O_OPENMP
# pragma omp parallel for default (shared) if (_numLandmarks > 100)
# endif
  for (int i = 0; i < _numLandmarks; ++i) {
    LandmarkMatrixType *b=_Hll->block(i,i);
    if (backup)
      _diagonalBackupLandmark[i] = b->diagonal();
    b->diagonal().array() += lambda;
  }
  return true;
}

template <typename Traits>
void BlockSolver<Traits>::restoreDiagonal()
{
  assert((int) _diagonalBackupPose.size() == _numPoses && "Mismatch in dimensions");
  assert((int) _diagonalBackupLandmark.size() == _numLandmarks && "Mismatch in dimensions");
  for (int i = 0; i < _numPoses; ++i) {
    PoseMatrixType *b=_Hpp->block(i,i);
    b->diagonal() = _diagonalBackupPose[i];
  }
  for (int i = 0; i < _numLandmarks; ++i) {
    LandmarkMatrixType *b=_Hll->block(i,i);
    b->diagonal() = _diagonalBackupLandmark[i];
  }
}

template <typename Traits>
bool BlockSolver<Traits>::init(SparseOptimizer* optimizer, bool online)
{
  _optimizer = optimizer;
  if (! online) {
    if (_Hpp)
      _Hpp->clear();
    if (_Hpl)
      _Hpl->clear();
    if (_Hll)
      _Hll->clear();
  }
  _linearSolver->init();
  return true;
}

template <typename Traits>
void BlockSolver<Traits>::setWriteDebug(bool writeDebug)
{
  _linearSolver->setWriteDebug(writeDebug);
}

template <typename Traits>
bool BlockSolver<Traits>::saveHessian(const std::string& fileName) const
{
  return _Hpp->writeOctave(fileName.c_str(), true);
}

template <typename Traits>
bool BlockSolver<Traits>::saveHessiansToFile(const std::string& fileNameStart) const
{

  // DEBUG(alexmillane)
  std::cout << "Saving hessians to file" << std::endl;
  std::cout << "Filename: " << fileNameStart << std::endl;

  // Filenames
  std::string fileName_Hpp(fileNameStart + "_Hpp");
  std::string fileName_Hll(fileNameStart + "_Hll");
  std::string fileName_Hpl(fileNameStart + "_Hpl");
  std::string fileName_Hschur(fileNameStart + "_Hschur");

  // Writing
  bool ok1 = true;
  bool ok2 = true;
  bool ok3 = true;
  bool ok4 = true;
  if(_Hpp)
    ok1 = _Hpp->writeMatlab(fileName_Hpp.c_str(), true);
  if (_Hll) 
    ok2 = _Hll->writeMatlab(fileName_Hll.c_str(), true);
  if (_Hll) 
    ok3 = _Hpl->writeMatlab(fileName_Hpl.c_str(), false);
  if (_Hll) 
    ok4 = _Hschur->writeMatlab(fileName_Hschur.c_str(), true);

  // Return
  return ok1 & ok2 & ok3 & ok4;
}

template <typename Traits>
bool BlockSolver<Traits>::computePoseCovariance(Eigen::MatrixXd& poseCovariance)
{
  // If not computing by schur's compliment this function will not work.
  if (!_doSchur)
    return false;
  // Timing
  double t=get_monotonic_time();
  // Restoring the diagonal in case Levenberg has fucked with it
  restoreDiagonal();
  // Recomputing the schur compliment with the original diagonal
  updateSchur();
  // Timing
  cerr << "Covariance [schur] = " <<  get_monotonic_time()-t << endl;
  // Retrieving the inverse of the schur compliment matrix
  poseCovariance.resize(_Hschur->rows(), _Hschur->cols());
  bool success = _linearSolver->solveInverse(*_Hschur, &poseCovariance);
  // Timing
  cerr << "Covariance [whole] = " <<  get_monotonic_time()-t << endl;
  // Success
  return success;
}

template <typename Traits>
bool BlockSolver<Traits>::computePartialPoseCovariance()
{
  // If not computing by schur's compliment this function will not work.
  if (!_doSchur)
    return false;

  // Timing
  std::cout << "Getting the schur compliment" << std::endl;
  double t=get_monotonic_time();
  // Restoring the diagonal in case Levenberg has fucked with it
  restoreDiagonal();
  // Recomputing the schur compliment with the original diagonal
  updateSchur();

  // Getting the cholesky factor
  // TODO(alexmillane): Returning a pure sparse matrix introduces a dependency on Eigen Sparse Matrix
  //                    up the entire chain. Would be good to move to block based reordering and return
  //                    a factor which is a sparse block matrix.
  std::cout << "Getting the cholesky factor" << std::endl;
  Eigen::SparseMatrix<double, Eigen::ColMajor> cholesky_factor;
  _linearSolver->getCholeskyFactor(*_Hschur, &cholesky_factor);

  // Getting the CCS representation of the cholesky factor
  std::cout << "Compressing the cholesky factor" << std::endl;
  cholesky_factor.makeCompressed();
  int n = cholesky_factor.rows();
  int* Lp = cholesky_factor.outerIndexPtr();
  int* Li = cholesky_factor.innerIndexPtr();
  double* Lx = cholesky_factor.valuePtr();

  // DEBUG
  // ----------------------------------------------
/*  // Inspecting the CCS elements
  for (size_t i = 0; i < cholesky_factor.nonZeros(); i++) {
    std::cout << "Lx: " << Lx[i] << std::endl;
  }

  for (size_t i = 0; i < cholesky_factor.nonZeros(); i++) {
    std::cout << "Li: " << Li[i] << std::endl;
  }

  for (size_t i = 0; i < cholesky_factor.cols(); i++) {
    std::cout << "Lp: " << Lp[i] << std::endl;
  }

  // Diagonal elements
  for (size_t i = 0; i < cholesky_factor.cols(); i++) {
    std::cout << "diag(cholesky_factor)[i]: " << cholesky_factor.coeffRef(i,i) << std::endl;
  }*/
  // ----------------------------------------------

  // Creating the marginal cholesky object
  std::cout << "Setting up the cholesky covariance solver" << std::endl;
  MarginalCovarianceCholesky marginal_covariance_cholesky;
  marginal_covariance_cholesky.setCholeskyFactor(n, Lp, Li, Lx, 0);

  // Saving the cholesky factor
  std::cout << "Saving the cholesky factor" << std::endl;
  std::string filename = "/home/millanea/Desktop/cholesky_factor";
  io::writeMatlab(filename, cholesky_factor);

  /* --------------------------------------------------
  
  // Testing out eigen's CCS storage
  /*
   * 0  3 0  0 0
   * 22 0 0  0 17
   * 7  5 0  1 0
   * 0  0 0  0 0
   * 0  0 14 0 8*/

  /*
  // Triplets list
  std::cout << "Creating the triplets list." << std::endl;
  std::vector<Eigen::Triplet<double>> triplet_list = {
      {0, 1, 3.0}, {1, 0, 22.0}, {1, 4, 17.0}, {2, 0, 7.0},
      {2, 1, 5.0}, {2, 3, 1.0},  {4, 2, 14.0}, {4, 4, 8.0}};
  for (const auto& triplet : triplet_list) {
    std::cout << "triplet: r: " << triplet.row() << ", c: " << triplet.col()
              << ", v: " << triplet.value() << std::endl;
  }

  // Creating the sparse matrix
  std::cout << "Creating the matrix from triplets list." << std::endl;
  Eigen::SparseMatrix<double, Eigen::ColMajor> test_sparse_matrix(5,5);
  test_sparse_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
  std::cout << "test_sparse_matrix: " << std::endl << test_sparse_matrix << std::endl;

  // Inspecting the CCS elements
  size_t nnz = test_sparse_matrix.nonZeros();
  test_sparse_matrix.makeCompressed();
  double* value_ptr = test_sparse_matrix.valuePtr();
  for (size_t i = 0; i < nnz; i++) {
    std::cout << "value: " << *value_ptr << std::endl;
    value_ptr++;
  }

  int* inner_index_ptr = test_sparse_matrix.innerIndexPtr();
  for (size_t i = 0; i < nnz; i++) {
    std::cout << "inner_index: " << *inner_index_ptr << std::endl;
    inner_index_ptr++;
  }

  int* outer_index_ptr = test_sparse_matrix.outerIndexPtr();
  for (size_t i = 0; i < test_sparse_matrix.cols(); i++) {
    std::cout << "outer_index: " << *outer_index_ptr << std::endl;
    outer_index_ptr++;
  }*/
}

} // end namespace
