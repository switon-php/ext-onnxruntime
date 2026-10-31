#!/usr/bin/env bash
# The extension links ONNX Runtime and system libraries. Anything else is a mistake.
set -euo pipefail

file=${1:?library path}
os=$(uname -s)

if [[ "$os" == Darwin ]]; then
  deps=$(otool -L "$file" | awk 'NR>1 { print $1 }')
  allowed='^(libonnxruntime|libc|libm|libgcc_s|libpthread|libdl|libc[+][+]|libiconv|libSystem)([.]|-|$)'
elif [[ "$os" == Linux ]]; then
  deps=$(readelf -d "$file" | awk '/\(NEEDED\)/ { gsub(/[][]/, "", $NF); print $NF }')
  allowed='^(libonnxruntime|libc|libm|libgcc_s|libpthread|libdl|ld-linux|linux-vdso|libgomp|libstdc[+][+]|libresolv|librt)([.]|-|$)'
elif [[ "$os" == MINGW* || "$os" == MSYS* || "$os" == CYGWIN* ]]; then
  readobj=""
  if [[ -n "${LIBCLANG_PATH:-}" ]]; then
    bindir=$(cygpath -u "$LIBCLANG_PATH")
    if [[ -f "$bindir/llvm-readobj.exe" ]]; then
      readobj="$bindir/llvm-readobj.exe"
    fi
  fi
  if [[ -z "$readobj" && -f "/c/Program Files/LLVM/bin/llvm-readobj.exe" ]]; then
    readobj="/c/Program Files/LLVM/bin/llvm-readobj.exe"
  fi
  if [[ -z "$readobj" ]]; then
    echo "llvm-readobj not found; cannot list DLL imports of $file" >&2
    exit 1
  fi
  imports=$(mktemp)
  set +e
  "$readobj" --coff-imports "$file" >"$imports"
  status=$?
  set -e
  deps=$(sed -n 's/^[[:space:]]*Name:[[:space:]]*\(.*\.[Dd][Ll][Ll]\)[[:space:]]*$/\1/p' "$imports")
  if [[ -z "$deps" ]]; then
    echo "llvm-readobj status $status produced no DLL imports for $file" >&2
    cat "$imports" >&2
    rm -f "$imports"
    exit 1
  fi
  rm -f "$imports"
  allowed='^(onnxruntime|php8|php8ts|kernel32|kernelbase|ntdll|bcrypt|bcryptprimitives|advapi32|ws2_32|userenv|secur32|crypt32|ncrypt|vcruntime140|vcruntime140_[0-9]+|msvcp140|msvcp140_[0-9]+|concrt140|ucrtbase|api-ms-win-.+)$'
else
  echo "unsupported OS $os" >&2
  exit 1
fi

shopt -s nocasematch
bad=0
saw_ort=0
while IFS= read -r dep; do
  [[ -z "$dep" ]] && continue
  base=$(basename "$dep")
  base=${base%.dll}
  base=${base%.DLL}
  if [[ "$base" == onnxruntime || "$base" == libonnxruntime* ]]; then
    saw_ort=1
  fi
  if [[ "$base" =~ $allowed ]]; then
    continue
  fi
  echo "unexpected library dependency: $dep" >&2
  bad=1
done <<<"$deps"
shopt -u nocasematch

if [[ "$saw_ort" -ne 1 ]]; then
  echo "extension does not link ONNX Runtime: $file" >&2
  bad=1
fi

if [[ "$bad" -ne 0 ]]; then
  echo "linkage of $file:" >&2
  if [[ "$os" == Darwin ]]; then
    otool -L "$file" >&2
  elif [[ "$os" == Linux ]]; then
    readelf -d "$file" >&2
  else
    echo "$deps" >&2
  fi
  exit 1
fi

echo "linkage ok: $file"
