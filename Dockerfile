# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: build the C++ engine and the Go server.
#
# CGO_ENABLED=1 is unavoidable — the server calls the engine through cgo — so
# the result is NOT a static binary. That is why the runtime stage below is
# debian-slim rather than scratch: it needs libc, libstdc++, libgomp, and the
# engine's own shared libraries.
# ---------------------------------------------------------------------------
FROM golang:1.26-bookworm AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Configure and build the engine first, in its own layer. The CMake step
# fetches and compiles llama.cpp, which is by far the slowest part of this
# build, so it is kept separate from the Go sources that change often.
COPY CMakeLists.txt ./
COPY include/ include/
COPY src/ src/
COPY tests/ tests/
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENGINE_OPENMP=ON \
    && cmake --build build --target engine -j"$(nproc)"

# Now the Go server. Module files first so dependency resolution caches.
COPY server/go.mod server/go.su[m] server/
WORKDIR /src/server
RUN go mod download

COPY server/ ./
# The rpath baked in by engine.go's cgo LDFLAGS points at the build directory,
# which will not exist in the runtime image. The libraries are installed into
# /usr/local/lib there instead, so that stale rpath is simply ignored.
RUN CGO_ENABLED=1 go build -trimpath -o /out/server .

# ---------------------------------------------------------------------------
# Stage 2: runtime.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libstdc++6 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# The engine plus llama.cpp/ggml, which it links dynamically. ldconfig makes
# them findable without LD_LIBRARY_PATH.
COPY --from=build /src/build/libengine.so /usr/local/lib/
COPY --from=build /src/build/bin/*.so* /usr/local/lib/
RUN ldconfig

COPY --from=build /out/server /usr/local/bin/server

# The model is NOT baked in: it is mounted at run time. See README for the
# tradeoff — a self-contained image is more portable, but a 469 MB layer makes
# every rebuild slow, and mounting guarantees that replicas on one node share
# the same page cache.
ENV MODEL=/models/qwen-2.5.gguf \
    ADDR=:8080 \
    WORKERS=1 \
    CONTEXT=1024

# Run unprivileged: nothing here needs root.
RUN useradd --system --uid 10001 --no-create-home engine
USER 10001

EXPOSE 8080

# Kubernetes probes these; see k8s/ in Phase 5.
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -fsS "http://localhost:8080/health/ready" || exit 1

# MODEL/ADDR/WORKERS/CONTEXT/THREADS above are read as defaults by the server,
# so this needs no arguments; flags still override them.
ENTRYPOINT ["/usr/local/bin/server"]
