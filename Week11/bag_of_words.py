import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown

def chunker(input_data, N):
  input_words = input_data.split(' ')
  output = [] 
  cur_chunk = []
  count = 0
  for word in input_words:
    cur_chunk.append(word)
    count += 1
    if count == N:
      output.append(' '.join(cur_chunk))
      count, cur_chunk = 0, []
  output.append(' '.join(cur_chunk))
  return output

input_data = ' '.join(brown.words()[:5400]) 
chunk_size = 800 
text_chunks = chunker(input_data, chunk_size) 
chunks = []
for count, chunk in enumerate(text_chunks):
  d = {'index': count, 'text': chunk}
  chunks.append(d) 

count_vectorizer = CountVectorizer(min_df=7, max_df=20)
document_term_matrix = count_vectorizer.fit_transform([chunk['text'] for chunk in chunks]) 
vocabulary = np.array(count_vectorizer.get_feature_names())
print("\nVocabulary:\n", vocabulary) 
chunk_names = []
for i in range(len(text_chunks)):
  chunk_names.append('Chunk-' + str(i+1)) 

print("\nDocument term matrix:")
formatted_text = '{:>12}' * (len(chunk_names) + 1)
print('\n', formatted_text.format('Word', *chunk_names), '\n')

for word, item in zip(vocabulary, document_term_matrix.T):
  output = [word] + [str(freq) for freq in item.data]
  print(formatted_text.format(*output)) 
