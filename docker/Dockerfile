# nix-llama container image.
#
# The point of building this with Nix rather than apt is parity: the
# llama-server binary in this image is the *same store path* the NixOS and
# nix-darwin services run, because both come from this flake's pinned nixpkgs.
# There is no second build to drift.
#
# Stage 1 builds the runtime closure with Nix. Stage 2 is `scratch` plus that
# closure -- nothing else is installed, so the image contains exactly what the
# server needs and nothing with its own patch cadence.
#
#   docker build -f docker/Dockerfile -t nix-llama:local .
#
# `nix build .#container` produces an equivalent image without Docker; CI
# builds both and checks they agree.

# hadolint ignore=DL3007
FROM nixos/nix:latest AS builder

# nix-command/flakes to build the flake at all; filter-syscalls=false because
# the kernel the build runs under may be older than the one Nix expects.
RUN mkdir -p /etc/nix \
    && printf 'experimental-features = nix-command flakes\nfilter-syscalls = false\nsandbox = false\n' \
       > /etc/nix/nix.conf

WORKDIR /src
COPY . /src

# The closure, the merged /bin, and the prepared /etc all come out of
# nix/container.nix -- the same expression `nix build .#container` uses.
RUN nix build .#container-closure --out-link /tmp/result \
    && mkdir -p /tmp/rootfs/nix/store \
    && xargs -a /tmp/result/store-paths cp -a -t /tmp/rootfs/nix/store \
    && cp -a "$(readlink -f /tmp/result/etc)/." /tmp/rootfs/ \
    && ln -s "$(readlink -f /tmp/result/root)/bin" /tmp/rootfs/bin \
    && mkdir -p /tmp/rootfs/var/lib/nix-llama \
    && chown -R 1000:1000 /tmp/rootfs/var/lib/nix-llama

FROM scratch

COPY --from=builder /tmp/rootfs /

# Kept in step with nix/container.nix by the `container-parity` flake check:
# changing one without the other fails CI.
ENV PATH=/bin \
    NIX_LLAMA_HOST=0.0.0.0 \
    NIX_LLAMA_STATE_DIR=/var/lib/nix-llama \
    NIX_LLAMA_MODEL_DIR=/model \
    HOME=/var/lib/nix-llama \
    LLAMA_ARG_OFFLINE=1

# Unprivileged by default. The server parses untrusted HTTP input; it has no
# reason to be root, and `--user` can still override this.
USER 1000:1000

WORKDIR /var/lib/nix-llama

EXPOSE 8080

# No shell in the image, so no shell-form healthcheck: nix-llama answers its
# own readiness question.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["/bin/nix-llama", "doctor"]

ENTRYPOINT ["/bin/nix-llama"]
CMD ["serve"]

LABEL org.opencontainers.image.title="nix-llama" \
      org.opencontainers.image.description="Turnkey llama.cpp server with hardware-aware defaults and documentation retrieval" \
      org.opencontainers.image.source="https://github.com/heywoodlh/nix-llama" \
      org.opencontainers.image.licenses="MIT"
