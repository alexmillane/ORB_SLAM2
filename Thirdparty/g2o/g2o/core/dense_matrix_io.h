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

#ifndef DENSE_MATRIX_IO
#define DENSE_MATRIX_IO

#include <fstream>
#include <iostream>
#include <iomanip>

#include <unsupported/Eigen/SparseExtra>

#include "eigen_types.h"

namespace g2o {
namespace io {

bool writeMatlab(const char* filename, const MatrixXD& mat) {
  std::string name = filename;
  std::string::size_type lastDot = name.find_last_of('.');
  if (lastDot != std::string::npos) 
    name = name.substr(0, lastDot);

  std::ofstream fout(filename);
  fout << std::setprecision(9) << std::fixed;

  // Writing to the file.
  fout << mat;

  return fout.good();
}

bool writeMatlab(const std::string& filename, const Eigen::SparseMatrix<double, Eigen::ColMajor>& mat) {
/*  std::string name = filename;
  std::string::size_type lastDot = name.find_last_of('.');
  if (lastDot != std::string::npos) 
    name = name.substr(0, lastDot);

  std::ofstream fout(filename);
  fout << std::setprecision(9) << std::fixed;

*/  

  return saveMarket(mat, filename);

/*  // Writing to the file.
  fout << mat;

  return fout.good();*/
}

} // end namespace
} // end namespace

#endif
