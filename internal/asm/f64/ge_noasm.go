// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !amd64 noasm appengine

package f64

// Ger performs the rank-one operation
//  A += alpha * x * y^T
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func Ger(m, n uintptr, alpha float64,
	x []float64, incX uintptr,
	y []float64, incY uintptr,
	a []float64, lda uintptr) {

	if incX == 1 && incY == 1 {
		x = x[:m]
		y = y[:n]
		for i, xv := range x {
			tmp := alpha * xv
			if tmp != 0 {
				atmp := a[i*lda : i*lda+n]
				AxpyUnitary(tmp, y, atmp)
			}
		}
		return
	}

	ix := kx
	for i := 0; i < m; i++ {
		tmp := alpha * x[ix]
		if tmp != 0 {
			AxpyInc(tmp, y, a[i*lda:i*lda+n], uintptr(n), uintptr(incY), 1, uintptr(ky), 0)
		}
		ix += incX
	}
}
