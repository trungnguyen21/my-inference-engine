package main

import "errors"

// errEngineBoom stands in for a failure coming out of the C++ engine.
var errEngineBoom = errors.New("tensor operation failed: boom")
