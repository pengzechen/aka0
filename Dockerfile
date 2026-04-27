FROM --platform=linux/amd64 ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install basic build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy and extract toolchains
WORKDIR /opt/toolchains

COPY Xuantie-900-gcc-elf-newlib-x86_64-V3.3.0-20260204.tar.gz /tmp/
COPY Xuantie-900-gcc-linux-6.6.36-glibc-x86_64-V3.3.0-20260204.tar.gz /tmp/
COPY Xuantie-900-gcc-linux-6.6.36-musl64-x86_64-V3.3.0-20260204.tar.gz /tmp/
COPY cvitek_tpu_sdk_cv181x_musl_riscv64_rvv\(1\).tar.gz /tmp/cvitek_tpu_sdk.tar.gz

RUN tar xzf /tmp/Xuantie-900-gcc-elf-newlib-x86_64-V3.3.0-20260204.tar.gz -C /opt/toolchains/ \
    && tar xzf /tmp/Xuantie-900-gcc-linux-6.6.36-glibc-x86_64-V3.3.0-20260204.tar.gz -C /opt/toolchains/ \
    && tar xzf /tmp/Xuantie-900-gcc-linux-6.6.36-musl64-x86_64-V3.3.0-20260204.tar.gz -C /opt/toolchains/ \
    && tar xzf /tmp/cvitek_tpu_sdk.tar.gz -C /opt/ \
    && rm /tmp/*.tar.gz

# Add all three toolchains to PATH
#   elf-newlib:     riscv64-unknown-elf-gcc / g++
#   linux-glibc:    riscv64-unknown-linux-gnu-gcc / g++
#   linux-musl64:   riscv64-unknown-linux-musl-gcc / g++
ENV PATH="/opt/toolchains/Xuantie-900-gcc-linux-6.6.36-glibc-x86_64-V3.3.0/bin:\
/opt/toolchains/Xuantie-900-gcc-linux-6.6.36-musl64-x86_64-V3.3.0/bin:\
/opt/toolchains/Xuantie-900-gcc-elf-newlib-x86_64-V3.3.0/bin:\
${PATH}"

# Verify toolchains on build
RUN echo "=== elf-newlib toolchain ===" \
    && riscv64-unknown-elf-gcc --version \
    && echo "=== linux-glibc toolchain ===" \
    && riscv64-unknown-linux-gnu-gcc --version \
    && echo "=== linux-musl64 toolchain ===" \
    && riscv64-unknown-linux-musl-gcc --version

WORKDIR /workspace
CMD ["/bin/bash"]
