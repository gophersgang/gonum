// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f64

import (
	"fmt"
	"testing"
)

func TestGer(t *testing.T) {
	tests := []struct {
		m, n uintptr
		x    []float64
	}{
		{11, 11, nil},
		{9, 9, nil},
		{9, 8, nil},
		{9, 4, nil},
		{9, 3, nil},
		{9, 2, nil},
		{9, 1, nil},

		{4, 4, nil},
		{3, 3, nil},

		{2, 1, nil},
		{2, 2, nil},
		{2, 3, nil},
		{2, 9, nil},

		{1, 1, nil},
		{1, 9, nil},
		{1, 2, nil},
		{1, 3, nil},
	}

	for i, test := range tests {
		test.x = []float64{0, 0}
		Ger(test.m, test.n, 0, test.x, 0, nil, 0, nil, 0)
		n := test.n * (test.m/4 + (test.m&2)>>1 + (test.m & 1))
		if float64(test.m) != test.x[0] || float64(n) != test.x[1] {
			t.Error(i, test.m, test.n, n, test.x)
			t.Error(test.m, test.n, test.m/4*test.n, test.n*((test.m&2)>>1), test.n*(test.m&1))
		}
		fmt.Println(test.m, test.n, test.x)
	}
}
