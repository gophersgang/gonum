// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.7

package f64

import (
	"fmt"
	"testing"

	"github.com/gonum/blas"
	"github.com/gonum/blas/testblas"
)

func TestGer(t *testing.T) {
	tests := []struct {
		m, n            uintptr
		alpha           float64
		x, y, a         []float64
		incX, incY, lda uintptr
		want            []float64
	}{
		{
			m: 1, n: 1, alpha: 1,
			x: []float64{2}, incX: 1,
			y: []float64{4.4}, incY: 1,
			a: []float64{10}, lda: 1,
			want: []float64{18.8},
		},
	}

	for _, test := range tests {
		Ger(test.m, test.n, test.alpha, test.x, test.incX, test.y, test.incY, test.a, test.lda)
	}
}

type dgerWrap struct{}

func (d dgerWrap) Dger(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int) {
	Ger(uintptr(m), uintptr(n), alpha, x, uintptr(incX), y, uintptr(incY), a, uintptr(lda))
}

func TestBlasGer(t *testing.T) {
	testblas.DgerTest(t, dgerWrap{})
}

func BenchmarkBlasGer(t *testing.B) {
	for _, dims := range newIncSet(3, 10, 30, 100, 300, 1000, 1e4, 1e5) {
		m, n := dims.x, dims.y
		if m/n >= 100 || n/m >= 100 || (m == 1e5 && n == 1e5) {
			continue
		}
		for _, inc := range newIncSet(1, 2, 3, 4, 10) {
			incX, incY := inc.x, inc.y
			t.Run(fmt.Sprintf("Dger %dx%d (%d %d)", m, n, incX, incY), func(b *testing.B) {
				for i := 0; i < t.N; i++ {
					testblas.DgerBenchmark(b, dgerWrap{}, m, n, incX, incY)
				}
			})

		}
	}
}

// The following are panic strings used during parameter checks.
const (
	negativeN = "blas: n < 0"
	zeroIncX  = "blas: zero x index increment"
	zeroIncY  = "blas: zero y index increment"
	badLenX   = "blas: x index out of range"
	badLenY   = "blas: y index out of range"

	mLT0  = "blas: m < 0"
	nLT0  = "blas: n < 0"
	kLT0  = "blas: k < 0"
	kLLT0 = "blas: kL < 0"
	kULT0 = "blas: kU < 0"

	badUplo      = "blas: illegal triangle"
	badTranspose = "blas: illegal transpose"
	badDiag      = "blas: illegal diagonal"
	badSide      = "blas: illegal side"

	badLdA = "blas: index of a out of range"
	badLdB = "blas: index of b out of range"
	badLdC = "blas: index of c out of range"

	badX = "blas: x index out of range"
	badY = "blas: y index out of range"
)

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

type dgemvWrap struct {
	t *testing.T
}

func (d dgemvWrap) Dgemv(tA blas.Transpose, m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(badTranspose)
	}
	if m < 0 {
		panic(mLT0)
	}
	if n < 0 {
		panic(nLT0)
	}
	if lda < max(1, n) {
		panic(badLdA)
	}

	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}
	// Set up indexes
	lenX := m
	lenY := n
	if tA == blas.NoTrans {
		lenX = n
		lenY = m
	}
	if (incX > 0 && (lenX-1)*incX >= len(x)) || (incX < 0 && (1-lenX)*incX >= len(x)) {
		panic(badX)
	}
	if (incY > 0 && (lenY-1)*incY >= len(y)) || (incY < 0 && (1-lenY)*incY >= len(y)) {
		panic(badY)
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic(badLdA)
	}

	// Quick return if possible
	if m == 0 || n == 0 || (alpha == 0 && beta == 1) {
		return
	}
	if alpha == 0 {
		if incY > 0 {
			ScalInc(beta, y, uintptr(lenY), uintptr(incY))
		} else {
			ScalInc(beta, y, uintptr(lenY), uintptr(-incY))
		}
		return
	}

	if tA == blas.NoTrans {
		GemvN(uintptr(m), uintptr(n), alpha, a, uintptr(lda), x, uintptr(incX), beta, y, uintptr(incY))
	} else {
		GemvT(uintptr(m), uintptr(n), alpha, a, uintptr(lda), x, uintptr(incX), beta, y, uintptr(incY))
	}
}

func TestBlasGemv(t *testing.T) {
	testblas.DgemvTest(t, dgemvWrap{t})
}
