INPUT_DIR = ./inputs
LOG_DIR = ./checkpoints
RUN_DIR = ./mlruns

all: $(INPUT_DIR) $(LOG_DIR) $(RUN_DIR)

$(INPUT_DIR):
	mkdir -p $(INPUT_DIR)

$(LOG_DIR):
	mkdir -p $(LOG_DIR)

$(RUN_DIR):
	mkdir -p $(RUN_DIR)
