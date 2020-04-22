all:
	echo "1/ make setup then 2/ make ngrams; or make lstm"

setup:
	conda env create -f environment.yml python=3.7.6

lstm:
	python3.7 -m chatbot-intents.lstm.train_lstm

ngrams:
	python3.7 -m chatbot-intents.ngrams.train_ngrams

clean:
	rm -f chatbot-intents/outputs/*.pkl

.PHONY: ngrams lstm
