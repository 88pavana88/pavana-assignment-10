# Variables
PYTHON = python
PIP = $(PYTHON) -m pip
ENV_DIR = venv
FLASK_APP = app.py

# Targets
.PHONY: all install run clean

# Default target
all: install run

# Set up the virtual environment and install dependencies
install:
	$(PYTHON) -m venv $(ENV_DIR)
	$(ENV_DIR)\Scripts\activate && $(PIP) install -r requirements.txt

# Run the Flask app
run:
	$(ENV_DIR)\Scripts\activate && $(PYTHON) $(FLASK_APP)

# Clean up generated files and environment
clean:
	@if exist $(ENV_DIR) (rmdir /s /q $(ENV_DIR))
	@if exist __pycache__ (rmdir /s /q __pycache__)
	@if exist *.pyc (del /q *.pyc)
