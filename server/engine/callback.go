package engine

// This file exists separately from engine.go for one cgo rule: a file
// containing //export must have a preamble with declarations only, never
// definitions. engine.go's preamble defines engine_run_bridge, so the exported
// function cannot live there.

/*
#include <stdint.h>
*/
import "C"

import "runtime/cgo"

// goEngineOnToken is called by the C engine once per decoded token piece,
// on whichever OS thread is running that request's generation.
//
//export goEngineOnToken
func goEngineOnToken(text *C.char, user C.uintptr_t) C.int {
	fn := cgo.Handle(uintptr(user)).Value().(TokenFunc)
	if fn(C.GoString(text)) {
		return 1 // keep going
	}
	return 0 // stop generating
}
