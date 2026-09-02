#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p go-server/internal/pb
protoc \
  --proto_path=proto \
  --go_out=go-server/internal/pb --go_opt=paths=source_relative \
  --go-grpc_out=go-server/internal/pb --go-grpc_opt=paths=source_relative \
  inference.proto

mkdir -p cpp-engine/generated
protoc \
  --proto_path=proto \
  --cpp_out=cpp-engine/generated \
  --grpc_out=cpp-engine/generated \
  --plugin=protoc-gen-grpc="$(which grpc_cpp_plugin)" \
  inference.proto

echo "Generated Go stubs in go-server/internal/pb/ and C++ stubs in cpp-engine/generated/"