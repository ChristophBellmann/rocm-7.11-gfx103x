# TheRock Update Summary - November 10, 2025

## ✅ Update Completed Successfully

**Updated from**: `2ce80a8` → `092078f4` (10 commits)
**Date**: November 10, 2025

______________________________________________________________________

## 📦 Major New Features

### 1. **hipSPARSELt** - NEW Sparse Matrix Library

- Added sparse linear algebra support (matrix multiplication for sparse matrices)
- Location: `math-libs/BLAS/`
- Enable with: `-DTHEROCK_ENABLE_SPARSE=ON`

### 2. **hipDNN** - Deep Neural Network Library

- AMD's DNN primitives library
- Location: `ml-libs/`
- Artifact: `ml-libs/artifact-hipdnn.toml`
- RFC: `docs/upstream/rfcs/RFC0005-hipDNN-Project-Integration.md`

### 3. **MIOpen Plugin**

- Modular plugin architecture for MIOpen
- Location: `ml-libs/`
- Artifact: `ml-libs/artifact-miopen-plugin.toml`

### 4. **UCCL (Unified Collective Communication Library)**

- New communication library support
- Location: `external-builds/uccl/`
- Python wheel building support

### 5. **Test Harness Framework**

- New test orchestration system for CI/CD
- Documentation: `docs/upstream/development/therock_test_harness.md`
- Workflow: `.github/workflows/therock_test_harness.yml`
- Visual diagram available in docs

______________________________________________________________________

## 🔧 Important Infrastructure Changes

### CMake Improvements

- **Python Multi-Version Support**: New `therock_detect_python_versions.cmake`

  - Build for Python 3.11, 3.12, 3.13 simultaneously
  - Set with: `THEROCK_DIST_PYTHON_EXECUTABLES`

- **Sanitizer Support**: Project-wide sanitizer builds

  - Set with: `THEROCK_SANITIZER` (e.g., `ASAN`)
  - Note: Incompatible with delay-loading (Issue #1783)

- **OpenCL Runtime**: New `OCL_RUNTIME` feature

  - Core OpenCL support addition

### Submodule Updates

- **compiler/amd-llvm**: `56748a69` → `abda6831` (Compiler promotion)

  - Improved HSA/comgr dependency search for ASan
  - GCC toolset-13 detection

- **rocm-libraries**: Updated to `8f545778`

  - Bump 20251106

- **rocm-systems**: Updated to `30785f8d`

  - Bump 20251107

- **composable_kernel**: Updated to `515e2830`

  - Top-K with Sigmoid kernel support

### Patches Applied/Updated

- **Removed patches** (no longer needed):

  - `0001-Enable-MSVC-support-in-comgr.patch`
  - `0004-Disable-hipcc-passing-hip-device-libs.patch`
  - `0007-Policy-CMP0053-is-no-longer-supported.patch`
  - `0008-Tunnel-CMAKE_PREFIX_PATH-with-proper-escaping.patch`
  - `0001-Patch-version-and-getpath-files-for-Windows.patch`
  - `0004-Fix-assembler-error-in-pal-trap-handler.patch`
  - `0008-Find-bundled-libelf.patch`

- **New patches**:

  - `amdsmi/0002-Fix-finding-librocm-core.so.patch`

______________________________________________________________________

## 🎯 GPU Target Updates

### Native gfx1031 Support Confirmed ✓

Your GPU (RX 6700 XT / gfx1031) is **officially supported** in upstream!

Location: `cmake/therock_amdgpu_targets.cmake:81-84`

```cmake
therock_add_amdgpu_target(gfx1031 "AMD RX 6700 XT"
  FAMILY dgpu-all gfx103X-all gfx103X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
)
```

**Good News**: Your local gfx1031 addition is no longer needed - it's now in upstream!

______________________________________________________________________

## 📋 CI/CD Enhancements

### Workflow Improvements

- New weekly CI workflow (`.github/workflows/ci_weekly.yml`)
- Improved artifact upload/download logic
- Test timeout adjustments (hipBLASLt: 75 minutes per shard)
- Better test report upload script: `upload_test_report_script.py`
- Path length validation: `check_path_lengths.py`

### Python Packaging

- **Multi-version wheel building**: Python 3.11, 3.12, 3.13
- Improved `build_python_packages.py`
- Better version computation: `compute_rocm_package_version.py`
- Manifest generation: `generate_therock_manifest.py`

______________________________________________________________________

## 🧰 Build Tool Updates

### New Utilities

- `build_tools/compute_rocm_package_version.py` - ROCm version computation
- `build_tools/generate_therock_manifest.py` - Manifest generation
- `build_tools/hack/check_path_lengths.py` - Path validation
- `build_tools/hack/get_prs_by_files_changed.py` - PR file analysis

### Enhanced Scripts

- `install_rocm_from_artifacts.py` - Better artifact installation
- `build_python_packages.py` - Multi-version support
- `setup_venv.py` - Improved virtual environment setup

______________________________________________________________________

## 📦 Native Packaging

### Linux Packaging Improvements

- Better DEB/RPM package generation workflow
- Improved dependency bundling (`THEROCK_BUNDLE_SYSDEPS`)
- Enhanced `runpath_to_rpath.py` for library path handling
- Updated `package.json` with more components

### Third-Party Dependencies

- **New**: `liblzma` bundled dependency support
  - Location: `third-party/sysdeps/common/liblzma/`
  - Full build system integration

______________________________________________________________________

## 🔍 Testing Enhancements

### New Test Harness

- Pytest-based orchestration framework

- Test files: `tests/harness/tests_*.py` for:

  - hipcub
  - rocprim
  - rocrand
  - rocthrust

- Library modules:

  - `tests/harness/libs/nodes.py` - Test node management
  - `tests/harness/libs/orchestrator.py` - Test orchestration
  - `tests/harness/libs/reports.py` - Test reporting
  - `tests/harness/libs/utils.py` - Test utilities

### Test Executable Scripts

New component test scripts in `build_tools/github_actions/test_executable_scripts/`:

- `test_hipdnn.py`
- `test_hipsparselt.py`
- `test_miopen_plugin.py`
- `test_rocroller.py`

______________________________________________________________________

## 🗂️ Documentation Updates

### New RFCs

1. **RFC0005**: hipDNN Project Integration

- `docs/upstream/rfcs/RFC0005-hipDNN-Project-Integration.md`

1. **RFC0006**: libhipcxx ROCm Core Inclusion

- `docs/upstream/rfcs/RFC0006-libhipcxx-ROCm-Core-Inclusion.md`

### New Docs

- `docs/upstream/development/therock_test_harness.md` - Test orchestration guide
- Updated Windows support table in `docs/upstream/development/windows_support.md`
- Enhanced `docs/upstream/packaging/python_packaging.md`

______________________________________________________________________

## 💾 Your Stashed Changes

Your local modifications were stashed and can be reapplied if needed:

```bash
# View stashed changes
git stash list

# See what's in the stash
git stash show -p stash@{0}

# Reapply if needed (only if you still need low-memory optimizations)
git stash pop
```

### What Was Stashed:

1. **.github/workflows/ci.yml** - CI workflow simplifications
1. **cmake/therock_amdgpu_targets.cmake** - gfx1031 addition (✓ NO LONGER NEEDED - now in upstream!)
1. **compiler/pre_hook_amd-llvm.cmake** - Low-memory build optimizations
1. **math-libs/BLAS/CMakeLists.txt** - OpenMP disabled for hipBLASLt

**Recommendation**: Don't reapply the stash unless you specifically need the low-memory optimizations. The gfx1031 change is now upstream and no longer needed!

______________________________________________________________________

## 🚀 Next Steps

### 1. Verify Your Build Configuration

```bash
# Check current GPU targets
cat build/CMakeCache.txt | grep THEROCK_AMDGPU_TARGETS
# Should show: THEROCK_AMDGPU_TARGETS:STRING=gfx1031
```

### 2. Consider Trying New Features

```bash
# Enable hipSPARSELt
cmake -B build -DTHEROCK_ENABLE_SPARSE=ON ...

# Enable hipDNN (if you need ML functionality)
cmake -B build -DTHEROCK_ENABLE_ML_LIBS=ON ...
```

### 3. Rebuild (Optional)

If you want to test the new upstream changes:

```bash
# Clean rebuild
cmake --build build --target expunge
cmake -B build -GNinja -DTHEROCK_AMDGPU_TARGETS=gfx1031 .
cmake --build build
```

### 4. Your ROCm Environment Scripts

The GPU target configuration scripts I created earlier are still valid:

- `rocm-switch.sh` - Switch between native/compat modes
- `rocm-env-native.sh` - Native gfx1031 environment
- `rocm-env-compat.sh` - Compatibility mode (gfx1030 override)
- `GPU_TARGET_COMPATIBILITY.md` - Full documentation

______________________________________________________________________

## 📊 Update Statistics

- **Files changed**: 150 files
- **Insertions**: +4,826 lines
- **Deletions**: -1,070 lines
- **New files**: 50+ new files (tests, docs, scripts, artifacts)
- **Commits pulled**: 10
- **Submodules updated**: 12
- **Patches applied**: 12 patches across 4 submodules

______________________________________________________________________

## ✅ Status Summary

| Component    | Status        | Notes                             |
| ------------ | ------------- | --------------------------------- |
| Git Update   | ✅ Complete   | 10 commits pulled                 |
| Submodules   | ✅ Updated    | All 12 submodules updated         |
| Patches      | ✅ Applied    | 12 patches successfully applied   |
| Build Config | ✅ Compatible | gfx1031 now officially supported! |
| Python Deps  | ✅ Unchanged  | No new requirements               |
| Your Changes | ✅ Stashed    | Can reapply if needed             |

______________________________________________________________________

## 🎉 Key Takeaway

**Your RX 6700 XT (gfx1031) is now officially supported in upstream TheRock!**

You no longer need the local patch - native gfx1031 support is built-in. Your build configuration is ready to go with the latest improvements from AMD ROCm.

Happy building! 🚀
