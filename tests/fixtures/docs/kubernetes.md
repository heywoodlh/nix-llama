# Kubernetes basics

## Deployments

A Deployment describes a desired state for a set of Pods and replaces them in a
controlled way when that state changes. `kubectl rollout status deployment/NAME`
reports progress; `kubectl rollout undo deployment/NAME` reverts it.

## Resource limits

A container's `resources.limits.memory` is enforced by the kernel through the
pod's cgroup. A process that exceeds it is killed, not throttled, so a model
server must be given enough headroom for its weights and its KV cache.
