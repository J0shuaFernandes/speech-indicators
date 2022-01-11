from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import audioread, librosa, pickle

def speech_to_text(audio):
	r = sr.Recognizer()
	with sr.AudioFile(audio) as source:
		# load audio to memory
		audio_data = r.record(source)
		# speech to text
		text = r.recognize_google(audio_data)
		return text.split(' ')

def no_of_pauses(audio):
	f = open('silence.pickle', 'rb')
	silence = pickle.load(f)
	f.close()

	wav, sr = librosa.load(audio)
	if sr != 22050:
		wav = librosa.resample(wav, sr, 22050)
	
	chunk_size = 4410
	prev, pauses = 1, 0	
	for i in range(0, len(wav), chunk_size):
		chunk = wav[i:i+chunk_size]
		if len(chunk) == 4410:	
			X = librosa.stft(chunk)
			Xdb = librosa.amplitude_to_db(abs(X))
			Xdb = Xdb.reshape(Xdb.shape[0]*Xdb.shape[1], -1)
			if cosine_similarity(silence.reshape(1,-1), Xdb.reshape(1,-1)) > 0.95:
				if prev == 1:
					pauses += 1
				prev = 0
			else:
				prev = 1
		else:
			pass
	return pauses

def unique_words(text):
	return len(set(text))

def repetition_of_words(text):
	words_unique = set(text)
	occurrences = [0 for x in range(len(words_unique))]
	words_dict = dict(zip(words_unique, occurrences))
	for x in words_unique:
		words_dict[x] = text.count(x)
	repeated_words = [x for x in words_unique if words_dict[x] > 1]
	return len(repeated_words)

def words_per_minute(audio, text):
	f = audioread.audio_open(audio)
	mins = (f.duration)/60
	wpm = len(text)/mins # no of words/duration in mins
	return round(wpm, 2) # round up to 2 decimal places

def count_interjections(audio): # aaa
	f = open('sample.pickle', 'rb')
	sample = pickle.load(f)
	f.close()
	wav, sr = librosa.load(audio)
	if sr != 22050:
		wav = librosa.resample(wav, sr, 22050)
	# split audio into chunks of 0.5 seconds
	count, chunk_size = 0, 11025
	for i in range(0, len(wav), chunk_size):
		chunk = wav[i:i+chunk_size]
		if len(chunk) == 11025:	
			X = librosa.stft(chunk)
			Xdb = librosa.amplitude_to_db(abs(X))
			Xdb = Xdb.reshape(Xdb.shape[0]*Xdb.shape[1], -1)
			if cosine_similarity(sample.reshape(1,-1), Xdb.reshape(1,-1)) > 0.9:
				count += 1
		else:
			pass

	return count

if __name__ == '__main__':
	file = input('Enter Filename: ')
	
	print('\nConverting speech to text...\n')
	text = speech_to_text(file)
	print('No. of Pauses: {}'.format(no_of_pauses(file)))
	print('Unique words: {}'.format(unique_words(text)))
	print('Words Per Minute: {}'.format(words_per_minute(file, text)))
	print('Repeated Words: {}'.format(repetition_of_words(text)))
	print('Interjections (aaa): {}'.format(count_interjections(file)))