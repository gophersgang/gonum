// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !noasm,!appengine

#include "textflag.h"

#define M_DIM m+0(FP)
#define M CX
#define N_DIM n+8(FP)
#define N BX
#define X_PTR SI
#define INC_X R8
#define INCx3_X R9
#define ALPHA X0
#define ALPHA_2 X1

#define LOAD4 \
	LONG  $0x0E120FF2               \ // MOVDDUP (SI), X1
	LONG  $0x120F41F2; WORD $0x3014 \ // MOVDDUP (SI)(R8), X2
	LONG  $0x120F42F2; WORD $0x461C \ // MOVDDUP (SI)(R8*2), X3
	LONG  $0x120F42F2; WORD $0x0E24 \ // MOVDDUP (SI)(R9*1), X4
	MULPD ALPHA, X1                 \
	MULPD ALPHA, X2                 \
	MULPD ALPHA, X3                 \
	MULPD ALPHA, X4                 \
	LEAQ  (SI)(INC_X*4), SI

#define LOAD2 \
	LONG  $0x0E120FF2               \ // MOVDDUP (SI), X1
	LONG  $0x120F41F2; WORD $0x3014 \ // MOVDDUP (SI)(R8), X2
	MULPD ALPHA, X1                 \
	MULPD ALPHA, X2                 \
	LEAQ  (SI)(INC_X*2), SI

#define LOAD1 \
	LONG  $0x06120FF2 \ // MOVDDUP (SI), X0
	MULPD ALPHA, X1   \
	ADDQ  INC_X, SI

// func Ger(m, n uintptr, alpha float64,
//	x []float64, incX uintptr,
//	y []float64, incY uintptr,
//	a []float64, lda uintptr)
TEXT ·Ger(SB), NOSPLIT, $0
	MOVQ M_DIM, M
	MOVQ N_DIM, N
	CMPQ M, $0
	JLE  end
	CMPQ N, $0
	JLE  end

	CMPQ $1, incY+80(FP) // Check for dense vector Y (fast-path)
	JG   inc
	JL   end

	MOVQ x+24(FP), X_PTR
	MOVQ incX+48(FP), INC_X

	SHRQ $2, M
	JZ   r2

r4:
	// LOAD 4
	LOAD4

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   r4c2

r4c4:
	// 4x4 KERNEL
	// STORE 4x4

	DECQ N
	JNZ  r4c4

r4c2:
	TESTQ $2, N_DIM
	JZ    r4c1

	// 4x2 KERNEL
	// STORE 4x2

r4c1:
	TESTQ $1, N_DIM
	JZ    r4end

	// 4x1 KERNEL
	// STORE 4x1

r4end:
	DECQ M
	JNZ  r4

r2:
	TESTQ $2, M_DIM
	JZ    r1

	// LOAD 2
	LOAD2

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   r2c2

r2c4:
	// 2x4 KERNEL
	// STORE 2x4

	DECQ N
	JNZ  r2c4

r2c2:
	TESTQ $2, N_DIM
	JZ    r2c1

	// 2x2 KERNEL
	// STORE 2x2

r2c1:
	TESTQ $1, N_DIM
	JZ    r1

	// 2x1 KERNEL
	// STORE 2x1

r1:
	TESTQ $1, M_DIM
	JZ    end

	// LOAD 1
	LOAD1

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   r1c2

r1c4:
	// 1x4 KERNEL
	// STORE 1x4

	DECQ N
	JNZ  r1c4

r1c2:
	TESTQ $2, N_DIM
	JZ    r1c1

	// 1x2 KERNEL
	// STORE 1x2

r1c1:
	TESTQ $1, N_DIM
	JZ    end

	// 1x1 KERNEL
	// STORE 1x1

end:
	RET

inc:  // Alogrithm for incY > 0 ( split loads in kernel )

	RET
