# Vendored Chronos

This directory vendors the Chronos GPU partitioner so Cumulus can run without a system-wide `chronos_cli`.

## Build (local)

1. Install build deps (cmake, a C++17 compiler, GPU SDK as required by your platform).

2. Build Chronos:

   ```bash
   cd cumulus/chronos_vendor/chronos
   mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j
   ```

3. Locate the CLI binary (usually at `build/apps/cli/chronos_cli`).

4. Either:

   - Export an env var to point Cumulus at the built binary:

     ```bash
     export CUMULUS_CHRONOS_PATH="$(pwd)/apps/cli/chronos_cli"
     ```

   - Or copy it under the expected vendored path:

     ```bash
     mkdir -p ../../bin
     cp apps/cli/chronos_cli ../../bin/chronos_cli
     ```

## Runtime Resolution Order

`cumulus.worker.CumulusManager` resolves the CLI path in this order:

1. `CUMULUS_CHRONOS_PATH` (if executable)
2. `cumulus/chronos_vendor/bin/chronos_cli` (if present)
3. System path `/usr/local/bin/chronos_cli`

## Notes

- Keep the Chronos LICENSE (Apache-2.0) intact (see `chronos/LICENSE`).
- If you prefer the system installation, skip the local build and ensure `chronos_cli` is in `/usr/local/bin`.
