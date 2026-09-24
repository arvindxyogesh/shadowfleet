import os

# Must run before `control_plane.app.config` is imported anywhere (its
# module-level `settings = Settings()` reads the environment at import
# time), so every test gets an isolated in-memory DB and never touches a
# real Redis instance. A service token is configured so the write
# endpoints (NFR-8) are reachable with the right X-Service-Token header.
os.environ.setdefault("SHADOWFLEET_CP_DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SHADOWFLEET_CP_REDIS_URL", "redis://localhost:1/0")
os.environ.setdefault("SHADOWFLEET_CP_SERVICE_TOKEN", "test-service-token")
