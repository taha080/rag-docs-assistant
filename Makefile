.PHONY: run test

run:
	uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

test:
	pytest
