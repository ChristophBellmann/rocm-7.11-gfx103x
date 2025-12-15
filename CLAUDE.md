# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TheRock (The HIP Environment and ROCm Kit) is a lightweight CMake super-project and build platform for HIP and ROCm. It provides:

- A unified build system for ROCm components using CMake
- Support for nightly releases of ROCm and PyTorch
- Cross-platform support (Linux distributions and native Windows)
- Comprehensive CI/CD pipelines for building, testing, and releasing components

TheRock is in early preview but under active development. It serves as both a CI/release platform and a development environment for ROCm component developers.

## Build System Architecture

TheRock uses a CMake-based super-project architecture where each ROCm component is a sub-project:

- **Super-Project**: TheRock itself, coordinating all sub-projects
- **Sub-Projects**: Individual ROCm components (e.g., amd-llvm, ROCR-Runtime, hipBLAS)
- **Build Phases**: Each sub-project goes through `configure > build > stage > dist` phases
- Sub-projects are organized into directories: `base/`, `compiler/`, `core/`, `comm-libs/`, `math-libs/`, `ml-libs/`, `profiler/`

### Build Directory Layout

Each sub-project creates:

- `build/`: CMake binary directory with build artifacts
- `stage/`: Isolated install directory for the component only
- `dist/`: Combined install with all runtime dependencies (uses hard-links from stage/)
- `stamp/`: Stamp files tracking build phase completion
- `_init.cmake`: Generated super-project initialization (sets up dependency resolver)
- `_toolchain.cmake`: Generated toolchain file (may use in-tree HIP compiler)

## Common Development Commands

### Initial Setup

```bash
# Ubuntu setup
sudo apt update
sudo apt install gfortran git ninja-build cmake g++ pkg-config xxd patchelf automake libtool python3-venv python3-dev libegl1-mesa-dev

# Clone and setup
git clone https://github.com/ROCm/TheRock.git
cd TheRock
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download submodules and apply patches
python3 ./build_tools/fetch_sources.py
```

### Building

```bash
# Basic build (must specify GPU target family or targets)
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu
cmake --build build

# With ccache (recommended for frequent rebuilds)
eval "$(./build_tools/setup_ccache.py)"
cmake -B build -GNinja -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  .
cmake --build build
```

### Building Subsets of Components

```bash
# Disable all, enable only specific components
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_HIPIFY=ON

# Reset features to defaults when changing configuration
cmake -B build -DTHEROCK_RESET_FEATURES=ON .
```

### Testing

```bash
# Enable testing during configure
cmake -B build -DBUILD_TESTING=ON ...

# Run tests
ctest --test-dir build
```

### Working with Sub-Projects

```bash
# Build specific component (with dependencies)
ninja -C build hipify

# Rebuild component from scratch
ninja -C build hipify+expunge && ninja -C build hipify

# Build only specific phase
ninja -C build hipify+configure
ninja -C build hipify+build
ninja -C build hipify+stage
ninja -C build hipify+dist

# Build part of tree using directory scoping
ninja -C build compiler/amd-llvm/all
ninja -C build math-libs/all
```

### Source Management

```bash
# Submodules are used for components
# To reset all submodules and reapply patches:
python3 ./build_tools/fetch_sources.py

# Work on individual components by cd'ing into their directories
# They are git submodules, so use normal git operations
```

### Code Assistance

```bash
# Generate combined compile_commands.json for IDE support
cmake --build build --target therock_merged_compile_commands
# Output: ./compile_commands.json
```

## Key Configuration Flags

### Required Flags

Must specify one of:

- `-DTHEROCK_AMDGPU_FAMILIES=<family>` (e.g., gfx110X-dgpu)
- `-DTHEROCK_AMDGPU_TARGETS=<targets>` (e.g., gfx1100)

See `cmake/therock_amdgpu_targets.cmake` for available options.

### Feature Group Flags

- `-DTHEROCK_ENABLE_ALL=OFF` - Disables all optional components
- `-DTHEROCK_ENABLE_CORE=OFF` - Disables core components
- `-DTHEROCK_ENABLE_COMM_LIBS=OFF` - Disables communication libraries
- `-DTHEROCK_ENABLE_MATH_LIBS=OFF` - Disables math libraries
- `-DTHEROCK_ENABLE_ML_LIBS=OFF` - Disables ML libraries
- `-DTHEROCK_ENABLE_PROFILER=OFF` - Disables profilers

### Individual Component Flags

- `-DTHEROCK_ENABLE_COMPILER=ON` - GPU+host compiler toolchain
- `-DTHEROCK_ENABLE_HIPIFY=ON` - hipify tool
- `-DTHEROCK_ENABLE_CORE_RUNTIME=ON` - Core runtime (ROCR-Runtime, Linux only)
- `-DTHEROCK_ENABLE_HIP_RUNTIME=ON` - HIP runtime
- `-DTHEROCK_ENABLE_RCCL=ON` - RCCL (Linux only)
- `-DTHEROCK_ENABLE_PRIM=ON` - PRIM libraries (rocprim, hipCUB, rocThrust)
- `-DTHEROCK_ENABLE_BLAS=ON` - BLAS libraries (hipblaslt, rocblas, hipblas)
- `-DTHEROCK_ENABLE_RAND=ON` - RAND libraries
- `-DTHEROCK_ENABLE_FFT=ON` - FFT libraries
- `-DTHEROCK_ENABLE_SOLVER=ON` - SOLVER libraries
- `-DTHEROCK_ENABLE_SPARSE=ON` - SPARSE libraries
- `-DTHEROCK_ENABLE_MIOPEN=ON` - MIOpen

### Developer-Focused Flags

- `-DCMAKE_BUILD_TYPE=<type>` - Build type (Release, Debug, RelWithDebInfo)
- `-D<project>_BUILD_TYPE=<type>` - Override build type for specific sub-project
- `-DTHEROCK_VERBOSE=ON` - Enable verbose CMake output
- `-DTHEROCK_BUNDLE_SYSDEPS=ON/OFF` - Use bundled system dependencies (default ON on Linux)
- `-DTHEROCK_ENABLE_MPI=OFF` - Build with MPI support
- `-DBUILD_TESTING=ON/OFF` - Enable/disable tests

### CMake Presets

See `CMakePresets.json` for predefined configurations:

- `linux-release-package` - RelWithDebInfo with split debug info
- `linux-release-asan` - RelWithDebInfo with AddressSanitizer
- `windows-release` - Release build for Windows

Use with: `cmake --preset linux-release-package -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu`

## Important Build Targets

### Top-Level Targets

- `all` (default) - Build all configured components
- `dist` - Build components and materialize unified `build/dist/rocm` tree
- `artifacts` - Generate all artifact directories and manifests
- `archives` - Generate `.tar.xz` archives
- `expunge` - Delete all sub-project build files
- `therock_merged_compile_commands` - Generate combined compile_commands.json

### Per-Component Targets

- `<component>` - Build component (runs all phases through dist)
- `<component>+configure` - Run configure phase only
- `<component>+build` - Run build phase only
- `<component>+stage` - Install to stage/ and dist/ directories
- `<component>+dist` - Generate artifacts that depend on component
- `<component>+expunge` - Remove all component intermediates

### Targets Within Component Build Directories

When working directly in `build/<path>/<component>/build/`:

- `therock-touch` - Mark component as needing re-staging
- `therock-dist` - Rebuild and update super-project dist/artifacts

## Key Python Scripts

- `build_tools/fetch_sources.py` - Download/reset submodules and apply patches
- `build_tools/setup_ccache.py` - Configure ccache for the project
- `build_tools/setup_venv.py` - Setup Python virtual environment with ROCm packages
- `build_tools/buildctl.py` - Build control utilities
- `build_tools/build_python_packages.py` - Build ROCm Python packages

## Pre-commit Hooks

The project uses pre-commit for automated checks:

```bash
pip install pre-commit

# Run on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Install as git hook
pre-commit install
```

Hooks include: trailing-whitespace, end-of-file-fixer, black (Python), clang-format (C/C++), mdformat (Markdown).

## Project Structure Notes

- Version information is in `version.json` (ROCm version) and `rocm-systems/projects/hip/VERSION` (HIP version)
- CMake modules are in `cmake/` directory
- Key CMake module: `cmake/therock_subproject.cmake` defines sub-project facilities
- External projects (e.g., PyTorch) are in `external-builds/`
- Patches for submodules are in `patches/`
- Build container definitions are in `dockerfiles/`

## Working with Component Build Directories

After initial super-project build, you can work directly in component build directories:

```bash
# Example: working on hipify
cd build/compiler/hipify/build
ninja
# Changes will mark the component as needing re-staging via therock-touch target

# To update super-project artifacts:
ninja therock-dist
```

The `_init.cmake` file in each component directory sets up:

- CMake dependency provider (redirects `find_package` to super-project deps)
- Program paths to find in-tree built tools
- Pre/post hooks for component-specific customization

## Platform-Specific Notes

### Linux

- Uses ROCR-Runtime (HSA runtime)
- Requires patchelf for bundled dependencies
- Supports native packaging (deb/rpm)

### Windows

- Uses PAL runtime (no separate core runtime)
- RCCL not supported
- Requires Visual Studio 2022
- Use `chcp 65001` for UTF-8 encoding on non-English systems

## Important Documentation

- `CONTRIBUTING.md` - Contribution guidelines and governance
- `docs/upstream/development/development_guide.md` - Detailed development workflow guide
- `docs/upstream/development/build_system.md` - Build system internals
- `docs/upstream/environment_setup_guide.md` - Environment setup for various platforms
- `docs/upstream/development/git_chores.md` - Version control procedures
- `RELEASES.md` - Release artifacts and installation
- `ROADMAP.md` - GPU support roadmap
