# Why Go for the serving layer

A short, plain explanation of why the HTTP layer in front of this engine is
written in Go. Written to be explainable to someone else in a minute or two.

## The situation

The inference engine is written in C++ from scratch — GGUF parsing, quantized
kernels, the transformer forward pass. That part is not changing.

What it needs is a way to be called over HTTP: a web server that accepts a
prompt, hands it to the engine, and streams the generated text back.

That web server does not need to be C++. It needs to be able to **call** C++.

## What we were choosing between

Python (FastAPI), Go, Java, and C#. All four can call C++ code. Java was ruled
out first — it needs either a very recent version (JDK 22+) or a lot of
hand-written glue code called JNI, and its runtime uses several times more
memory than the alternatives, which matters on an 8 GB machine.

That left Go and C#. They are genuinely close. Here is why Go won.

## Why Go won

**1. It matches how the engine is already written.**

The engine is C-style C++: functions return an integer success flag and write
error text into a buffer the caller provides. No exceptions, no classes.

```c
int engine_forward_token(..., char *error, int error_capacity);
```

Go works the same way — every call returns a value and an error, checked right
there:

```go
result, err := engine.Generate(prompt)
if err != nil { ... }
```

C# would push the opposite direction: exceptions instead of return codes,
classes and interfaces, and `async`/`await` throughout. That is three unfamiliar
paradigms at once for someone coming from C.

**2. Concurrency is much simpler to reason about.**

Serving several requests at once is the core of this project. In Go, running
something concurrently is one keyword, and passing results between those pieces
is a channel:

```go
go handleRequest(w, r)       // run concurrently
tokens := make(chan string)  // pass data between them
```

C#'s equivalent is `async`/`await`, which is powerful but has failure modes —
deadlocks, thread-pool starvation — that are invisible until they bite.

**3. It is the language of this ecosystem.**

Kubernetes, Docker, etcd, Prometheus, and containerd are all written in Go. For
a project about deployment and distributed systems, Go is the native tongue of
the tools being used.

**4. It is light, and ships as one file.**

A Go web server uses roughly 15 MB of memory, against ~70 MB for C# (or ~30 MB
with extra build effort) and 150–300 MB for Java. Running four copies on an
8 GB laptop, that difference is real. Go also compiles to a single executable,
which keeps container images small and startup fast.

**5. The language is small.**

About 25 keywords, usually one obvious way to do things. C# has accumulated
20+ years of features; you don't need all of them, but you meet them in every
tutorial and it is hard to tell the essential from the incidental.

## The honest caveats

- **Calling C++ is fiddlier in Go than in C#.** Go uses a mechanism called cgo,
  where you manage memory across the language boundary by hand. C#'s equivalent
  is cleaner. The tradeoff was accepted deliberately: that code is ~150 lines
  written once, while the concurrency and HTTP code — where Go is easier — is
  what gets changed repeatedly.
- **cgo gives up one of Go's nicest properties.** Normally Go cross-compiles to
  any platform trivially; with cgo you need a C toolchain for the target.
- **C# was a close second**, and would have been fine. Java was the only option
  ruled out on the merits rather than on preference.
- **Python with FastAPI would have been fastest to write**, at a cost in memory
  and type safety.

## One-sentence version

> Go matches the C-style error handling the engine already uses, makes
> concurrency simple enough to get right, and is the language Kubernetes and
> Docker are written in — and the one thing C# does better, calling C++, is a
> small amount of code written once.
