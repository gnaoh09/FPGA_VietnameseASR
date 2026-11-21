#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
PROJ_NAME="${PROJ_NAME:-top}"
TOP_MODULE="${TOP_MODULE:-top}"
ENG_MHZ="${ENG_MHZ:-80}"
SDRAM_MHZ="${SDRAM_MHZ:-100}"
MAX_T_TILE="${MAX_T_TILE:-4}"
CLK1_PHASE_DEG="${CLK1_PHASE_DEG:-0}"
QUARTUS_IMAGE="${QUARTUS_IMAGE:-raetro/quartus:21.1.1}"
BOARD_DIR="${ROOT_DIR}/boards/omdazz_ep4ce10"

mkdir -p "${BUILD_DIR}"
cp "${BOARD_DIR}/omdazz_ep4ce10.sdc" "${BUILD_DIR}/omdazz_ep4ce10.sdc"
cp "${BOARD_DIR}/omdazz_ep4ce10.qsf" "${BUILD_DIR}/${PROJ_NAME}.qsf"

cat > "${BUILD_DIR}/${PROJ_NAME}.qpf" <<EOF
PROJECT_REVISION = "${PROJ_NAME}"
EOF

{
  echo ""
  echo "# Auto-discovered RTL sources"
  echo "set_global_assignment -name TOP_LEVEL_ENTITY ${TOP_MODULE}"
  echo "set_parameter -name ENG_MHZ ${ENG_MHZ}"
  echo "set_parameter -name SDRAM_MHZ ${SDRAM_MHZ}"
  echo "set_parameter -name MAX_T_TILE ${MAX_T_TILE}"
  echo "set_parameter -name CLK1_PHASE_DEG ${CLK1_PHASE_DEG}"
  for src in "${ROOT_DIR}"/rtl/*.v "${ROOT_DIR}"/rtl/*.sv; do
    [ -e "${src}" ] || continue
    rel="../rtl/$(basename "${src}")"
    if [[ "${src}" == *.sv ]]; then
      echo "set_global_assignment -name SYSTEMVERILOG_FILE ${rel}"
    else
      echo "set_global_assignment -name VERILOG_FILE ${rel}"
    fi
  done
} >> "${BUILD_DIR}/${PROJ_NAME}.qsf"

docker run --rm --platform linux/amd64 \
  -v "${ROOT_DIR}:/work" \
  -w /work/build \
  "${QUARTUS_IMAGE}" \
  quartus_sh --flow compile "${PROJ_NAME}"

docker run --rm --platform linux/amd64 \
  -v "${ROOT_DIR}:/work" \
  -w /work/build \
  "${QUARTUS_IMAGE}" \
  quartus_pow "${PROJ_NAME}" || true

RBF_PATH="${BUILD_DIR}/output_files/${PROJ_NAME}.rbf"
SOF_PATH="${BUILD_DIR}/output_files/${PROJ_NAME}.sof"
SVF_PATH="${BUILD_DIR}/output_files/${PROJ_NAME}.svf"

if [ -f "${SOF_PATH}" ]; then
  docker run --rm --platform linux/amd64 \
    -v "${ROOT_DIR}:/work" \
    -w /work/build/output_files \
    "${QUARTUS_IMAGE}" \
    quartus_cpf -c -q 6MHz -g 3.3 -n p "${PROJ_NAME}.sof" "${PROJ_NAME}.svf"
fi

echo ""
echo "=== Resource & Timing Summary ==="
cat "${BUILD_DIR}/output_files/${PROJ_NAME}.fit.summary" 2>/dev/null || true
cat "${BUILD_DIR}/output_files/${PROJ_NAME}.sta.summary" 2>/dev/null || true
ls -lh "${SOF_PATH}" "${SVF_PATH}" "${RBF_PATH}" 2>/dev/null || true
