from tenserflow.keras.layers import textVectorization
x = df['comment_text']
y = df[df.columns[2:]].values

MAX_FEATURES = 200000
vectorizer = textVectorization(max_tokens = MAX_FEATURES,output_sequence_length=1800,output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

dataset = tf.data.dataset.from_tensor_slices((vectorized_text,y))
dataset = dataset.cache()
datast = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)
batch_X, batch_Y = dataset.as_numpy_iterator().next()

train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
train = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

train_generator = train.as_numpy_iterator()
train_generator.next()





