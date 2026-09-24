.PHONY: up down logs export-model demo test secret-scan

up:
	docker compose -f infra/docker-compose.yml up --build -d

down:
	docker compose -f infra/docker-compose.yml down

logs:
	docker compose -f infra/docker-compose.yml logs -f

export-model:
	pip install ultralytics
	python edge_agent/scripts/export_model.py --output edge_agent/models/yolov8n.onnx

demo:
	pip install -r scripts/requirements.txt
	python scripts/demo.py

test:
	pytest edge_agent/tests control_plane/tests training_pipeline/tests -v

# NFR-10: same scan CI runs (.github/workflows/secret-scan.yml). Needs a
# local gitleaks install: https://github.com/gitleaks/gitleaks#installing
secret-scan:
	gitleaks git . --redact --verbose --no-banner
