#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# setup.sh — One-shot install for the reconstruction pipeline
# Idempotent: safe to run multiple times.
# Target: Ubuntu container with optional AMD ROCm GPU
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-/workspace/SpatialBuild}"
CONFIG_FILE="$SCRIPT_DIR/.pipeline_config"
OPENSPLAT_DIR="$WORKSPACE/OpenSplat"
LIBTORCH_DIR="$WORKSPACE/libtorch"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# ── Detect ROCm ──────────────────────────────────────────────────────────────
ROCM_AVAILABLE=0
ROCM_ARCH=""
if [ -d "/opt/rocm" ] && [ -x "/opt/rocm/bin/rocminfo" ]; then
    ROCM_AVAILABLE=1
    ROCM_ARCH=$(/opt/rocm/bin/rocminfo 2>/dev/null \
        | grep -oP 'gfx\d+' | head -1 || echo "")
    info "ROCm detected at /opt/rocm  (arch: ${ROCM_ARCH:-unknown})"
else
    warn "ROCm not found — will build OpenSplat for CPU only"
fi

# ── 1. System packages (COLMAP) ─────────────────────────────────────────────
info "Checking COLMAP …"
if command -v colmap &>/dev/null; then
    info "COLMAP already installed: $(colmap -h 2>&1 | head -1 || echo 'OK')"
else
    info "Installing COLMAP via apt …"
    sudo apt-get update -qq
    sudo apt-get install -y -qq colmap
    if command -v colmap &>/dev/null; then
        info "COLMAP installed successfully"
    else
        fail "COLMAP installation failed — install manually"
    fi
fi

# ── 2. Python dependencies ───────────────────────────────────────────────────
info "Installing Python packages …"
pip install --break-system-packages -q \
    numpy Pillow plyfile requests trimesh 2>/dev/null \
    || pip install -q numpy Pillow plyfile requests trimesh

# ── 3. OpenSplat ─────────────────────────────────────────────────────────────
OPENSPLAT_BIN=""

build_opensplat() {
    local mode="$1"  # "rocm" or "cpu"

    info "Building OpenSplat ($mode) …"
    mkdir -p "$OPENSPLAT_DIR/build"

    # ── Get LibTorch if missing ──
    if [ ! -d "$LIBTORCH_DIR" ]; then
        info "Downloading LibTorch …"
        local torch_url
        if [ "$mode" = "rocm" ]; then
            local rocm_ver
            rocm_ver=$(cat /opt/rocm/.info/version 2>/dev/null | cut -d. -f1-2 || echo "6.0")
            torch_url="https://download.pytorch.org/libtorch/rocm${rocm_ver}/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Brocm${rocm_ver}.zip"
            # Fallback to CPU torch if ROCm-specific URL fails
            wget -q --spider "$torch_url" 2>/dev/null || \
                torch_url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
        else
            torch_url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
        fi
        (cd "$WORKSPACE" && wget -q "$torch_url" -O libtorch.zip && unzip -qo libtorch.zip && rm -f libtorch.zip) || {
            fail "LibTorch download failed"; return 1
        }
    fi

    # ── Clone repo if missing ──
    if [ ! -d "$OPENSPLAT_DIR/.git" ]; then
        git clone --depth 1 https://github.com/pierotofy/OpenSplat.git "$OPENSPLAT_DIR"
    fi

    cd "$OPENSPLAT_DIR/build"

    if [ "$mode" = "rocm" ]; then
        export PYTORCH_ROCM_ARCH="${ROCM_ARCH:-gfx942}"
        cmake .. \
            -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
            -DGPU_RUNTIME="HIP" \
            -DHIP_ROOT_DIR=/opt/rocm \
            -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON \
            2>&1 | tail -5
        make -j"$(nproc)" 2>&1 | tail -5
    else
        cmake .. \
            -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
            -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON \
            2>&1 | tail -5
        make -j"$(nproc)" 2>&1 | tail -5
    fi

    if [ -f "$OPENSPLAT_DIR/build/opensplat" ]; then
        OPENSPLAT_BIN="$OPENSPLAT_DIR/build/opensplat"
        info "OpenSplat binary: $OPENSPLAT_BIN"
        return 0
    else
        fail "OpenSplat build produced no binary"
        return 1
    fi
}

# Skip build if binary already exists
if [ -f "$OPENSPLAT_DIR/build/opensplat" ]; then
    OPENSPLAT_BIN="$OPENSPLAT_DIR/build/opensplat"
    info "OpenSplat already built: $OPENSPLAT_BIN"
elif [ "$ROCM_AVAILABLE" -eq 1 ]; then
    build_opensplat "rocm" || {
        warn "ROCm build failed — falling back to CPU build"
        rm -rf "$OPENSPLAT_DIR/build"
        build_opensplat "cpu" || fail "OpenSplat build failed entirely"
    }
else
    build_opensplat "cpu" || fail "OpenSplat build failed"
fi

# ── 4. Write config ─────────────────────────────────────────────────────────
cat > "$CONFIG_FILE" <<EOF
OPENSPLAT_BIN=${OPENSPLAT_BIN}
ROCM_AVAILABLE=${ROCM_AVAILABLE}
ROCM_ARCH=${ROCM_ARCH}
WORKSPACE=${WORKSPACE}
EOF
info "Config written to $CONFIG_FILE"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo " Setup Summary"
echo "══════════════════════════════════════════════"
echo -e " COLMAP:      $(command -v colmap &>/dev/null && echo "${GREEN}installed${NC}" || echo "${RED}MISSING${NC}")"
echo -e " OpenSplat:   $([ -n "$OPENSPLAT_BIN" ] && echo "${GREEN}$OPENSPLAT_BIN${NC}" || echo "${RED}NOT BUILT${NC}")"
echo -e " ROCm:        $([ "$ROCM_AVAILABLE" -eq 1 ] && echo "${GREEN}yes (${ROCM_ARCH})${NC}" || echo "${YELLOW}no (CPU mode)${NC}")"
echo -e " Python pkgs: $(python3 -c 'import numpy,PIL,plyfile,requests,trimesh; print("all OK")' 2>/dev/null || echo "${RED}MISSING — run pip install -r requirements.txt${NC}")"
echo "══════════════════════════════════════════════"
