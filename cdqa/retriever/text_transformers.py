# BSD 3-Clause License
#
# Copyright (c) 2018, Sho IIZUKA
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
from sklearn.feature_extraction.text import _document_frequency
from sklearn.preprocessing import normalize


class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
     norm : 'l1', 'l2' or None, optional (default=None)
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine
        similarity between two vectors is their dot product when l2 norm has
        been applied.
        * 'l1': Sum of absolute values of vector elements is 1.
    use_idf : boolean, optional (default=True)
        Enable inverse-document-frequency reweighting
    k1 : float, optional (default=2.0)
        term k1 in the BM25 formula
    b : float, optional (default=0.75)
        term b in the BM25 formula
    floor : float or None, optional (default=None)
        floor value for idf terms
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """

    def __init__(self, norm=None, use_idf=True, k1=2.0, b=0.75, floor=None):
        self.norm = norm
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b
        self.floor = floor

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        """
        X = check_array(X, accept_sparse=("csr", "csc"))
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            if self.floor is not None:
                idf = idf * (idf > self.floor) + self.floor * (idf < self.floor)
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)

        # Create BM25 features

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        dl = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]
        # In each row, repeat `dl` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(dl), sz)
        # Average document length
        # Scalar value
        avgdl = np.average(dl)
        # Compute BM25 score only for non-zero elements
        data = (
            X.data
            * (self.k1 + 1)
            / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
        )
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        self._doc_matrix = X
        return self

    def transform(self, X=None, copy=True, is_query=False):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term query matrix
        copy : boolean, optional (default=True)
        query: boolean (default=False)
            whether to transform a query or the documents database

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]

        """
        if is_query:
            X = check_array(X, accept_sparse="csr", dtype=FLOAT_DTYPES, copy=copy)
            if not sp.issparse(X):
                X = sp.csr_matrix(X, dtype=np.float64)

            n_samples, n_features = X.shape

            expected_n_features = self._doc_matrix.shape[1]
            if n_features != expected_n_features:
                raise ValueError(
                    "Input has n_features=%d while the model"
                    " has been trained with n_features=%d"
                    % (n_features, expected_n_features)
                )

            if self.use_idf:
                check_is_fitted(self, "_idf_diag", "idf vector is not fitted")
                X = sp.csr_matrix(X.toarray() * self._idf_diag.diagonal())

            return X

        else:
            return self._doc_matrix

    @property
    def idf_(self):
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))

    @idf_.setter
    def idf_(self, value):
        value = np.asarray(value, dtype=np.float64)
        n_features = value.shape[0]
        self._idf_diag = sp.spdiags(
            value, diags=0, m=n_features, n=n_features, format="csr"
        )
