all:
	echo "ready"

lstm:
	python3.7 -m chatbot-intents.lstm.train_lstm

ngrams:
	python3.7 -m chatbot-intents.ngrams.train_ngrams

clean:
	rm -f chatbot-intents/outputs/*.pkl

.PHONY: ngrams lstm
