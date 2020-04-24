all:
	echo "1/ bash configure 2/ make setup then 3/ make ngrams; or make lstm"

setup:
	conda env create -f environment.yml python=3.7.6

clean:
	rm -f chatbot-intents/outputs/*.pkl

activate:
	source /opt/anaconda3/etc/profile.d/conda.sh && conda activate tf2

ngrams-english: activate
	python3.7 -m chatbot-intents.ngrams.train_ngrams english

lstm-english: activate
	python3.7 -m chatbot-intents.lstm.train_lstm english

ngrams-french: activate
	python3.7 -m chatbot-intents.ngrams.train_ngrams french

lstm-french: activate
	python3.7 -m chatbot-intents.lstm.train_lstm french

ngrams-russian: activate
	python3.7 -m chatbot-intents.ngrams.train_ngrams russian

lstm-russian: activate
	python3.7 -m chatbot-intents.lstm.train_lstm russian

ngrams-german: activate
	python3.7 -m chatbot-intents.ngrams.train_ngrams german

lstm-german: activate
	python3.7 -m chatbot-intents.lstm.train_lstm german

ngrams-italian: activate
	python3.7 -m chatbot-intents.ngrams.train_ngrams italian

lstm-italian: activate
	python3.7 -m chatbot-intents.lstm.train_lstm italian

ngrams-spanish: activate
	python3.7 -m chatbot-intents.ngrams.train_ngrams spanish

lstm-spanish: activate
	python3.7 -m chatbot-intents.lstm.train_lstm spanish

