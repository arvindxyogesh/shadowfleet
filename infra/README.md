# infra

Deployment and CI/CD glue: Docker Compose for local multi-node fleet
simulation, optional k3d/kind manifests for a local-Kubernetes variant, and
GitHub Actions workflows for tests, retrain triggers, and the Hugging Face
Spaces demo deploy.

See `../docs/SRS.md` §2.4 (operating environment) and §6.7 (cost constraint —
everything here must stay within free tiers).
