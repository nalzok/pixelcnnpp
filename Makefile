.PHONY: run

run:
	pipenv run python main.py --workdir=workdir --config=configs/default.py --config.batch_size=320
