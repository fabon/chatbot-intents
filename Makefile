all:
	echo "1/ bash configure 2/ make setup then 3/ make ngrams; or make lstm"

setup:
	conda env create -f environment.yml python=3.7.6

clean:
	rm -f chatbot-intents/outputs/*.pkl

ngrams-english:
	python3.7 -m chatbot-intents.ngrams.train_ngrams english

lstm-english:
	python3.7 -m chatbot-intents.lstm.train_lstm english

ngrams-french:
	python3.7 -m chatbot-intents.ngrams.train_ngrams french

lstm-french:
	python3.7 -m chatbot-intents.lstm.train_lstm french

ngrams-russian:
	python3.7 -m chatbot-intents.ngrams.train_ngrams russian

lstm-russian:
	python3.7 -m chatbot-intents.lstm.train_lstm russian

ngrams-german:
	python3.7 -m chatbot-intents.ngrams.train_ngrams german

lstm-german:
	python3.7 -m chatbot-intents.lstm.train_lstm german

ngrams-italian:
	python3.7 -m chatbot-intents.ngrams.train_ngrams italian

lstm-italian:
	python3.7 -m chatbot-intents.lstm.train_lstm italian

ngrams-spanish:
	python3.7 -m chatbot-intents.ngrams.train_ngrams spanish

lstm-spanish:
	python3.7 -m chatbot-intents.lstm.train_lstm spanish

