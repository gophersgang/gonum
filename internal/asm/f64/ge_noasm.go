// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !go1.8 !amd64 noasm appengine

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
			AxpyUnitary(alpha*xv, y, a[uintptr(i)*lda:uintptr(i)*lda+n])
		}
		return
	}

	var ky, kx uintptr
	if incY > 0 {
		ky = 0
	} else {
		ky = -(n - 1) * incY
	}
	if incX > 0 {
		kx = 0
	} else {
		kx = -(m - 1) * incX
	}

	ix := kx
	for i := 0; i < int(m); i++ {
		AxpyInc(alpha*x[ix], y, a[uintptr(i)*lda:uintptr(i)*lda+n], uintptr(n), uintptr(incY), 1, uintptr(ky), 0)
		ix += incX
	}
}

// GemvN computes
//  y = alpha * a * x + beta * y
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func GemvN(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	var kx, ky, i uintptr
	if int(incX) < 0 {
		kx = uintptr(-int(n-1) * int(incX))
	}
	if int(incY) < 0 {
		ky = uintptr(-int(m-1) * int(incY))
	}

	if incX == 1 && incY == 1 {
		for i = 0; i < m; i++ {
			y[i] = y[i]*beta + alpha*DotUnitary(a[lda*i:lda*i+n], x)
		}
		return
	}
	iy := ky
	for i = 0; i < m; i++ {
		y[iy] = y[iy]*beta + alpha*DotInc(x, a[lda*i:lda*i+n], n, incX, 1, kx, 0)
		iy += incY
	}
}

// GemvT computes
//  y = alpha * A^T * x + beta * y
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func GemvT(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	var kx, ky, i uintptr
	if int(incX) < 0 {
		kx = uintptr(-int(m-1) * int(incX))
	}
	if int(incY) < 0 {
		ky = uintptr(-int(n-1) * int(incY))
		ScalInc(beta, y, n, uintptr(int(-incY)))
	} else if incY == 1 {
		ScalUnitary(beta, y)
	} else {
		ScalInc(beta, y, n, incY)
	}

	if incX == 1 && incY == 1 {
		for i = 0; i < m; i++ {
			tmp := alpha * x[i]
			if tmp != 0 {
				AxpyUnitaryTo(y, tmp, a[lda*i:lda*i+n], y)
			}
		}
		return
	}
	ix := kx
	for i = 0; i < m; i++ {
		tmp := alpha * x[ix]
		if tmp != 0 {
			AxpyInc(tmp, a[lda*i:lda*i+n], y, n, 1, incY, 0, ky)
		}
		ix += incX
	}

}
