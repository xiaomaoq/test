import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# load dataset
# ['evaluation'] is feature, ['label'] is label
def load_data(filepath, input_shape=20):
    df = pd.read_csv(filepath, encoding='gbk')

    # 标签及词汇表
    labels, vocabulary = list(df['label'].unique()), list(df['evaluation'].unique())

    # 构造字符级别的特征
    string = ''
    for word in vocabulary:
        string += word

    vocabulary = set(string)

    # 字典列表
    word_dictionary = {word: i + 1 for i, word in enumerate(vocabulary)}
    with open('word_dict.pk', 'wb') as f:
        pickle.dump(word_dictionary, f)
    inverse_word_dictionary = {i + 1: word for i, word in enumerate(vocabulary)}
    label_dictionary = {label: i for i, label in enumerate(labels)}
    with open('label_dict.pk', 'wb') as f:
        pickle.dump(label_dictionary, f)
    output_dictionary = {i: labels for i, labels in enumerate(labels)}

    # 词汇表大小
    vocab_size = len(word_dictionary.keys())
    # 标签类别数量
    label_size = len(label_dictionary.keys())

    # 序列填充，按input_shape填充，长度不足的按0补充
    x = [[word_dictionary[word] for word in sent] for sent in df['evaluation']]
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    y = [[label_dictionary[sent]] for sent in df['label']]
    '''
    to_categorical用于将标签转化为形如(nb_samples, nb_classes)
    的二值序列。
    假设num_classes = 10。
    如将[1, 2, 3,……4]转化成：
    [[0, 1, 0, 0, 0, 0, 0, 0]
     [0, 0, 1, 0, 0, 0, 0, 0]
     [0, 0, 0, 1, 0, 0, 0, 0]
    …… 
    [0, 0, 0, 0, 1, 0, 0, 0]]
    '''
    y = [to_categorical(label, num_classes=label_size) for label in y]
    y = np.array([list(_[0]) for _ in y])

    return x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary


# 创建深度学习模型，Embedding + LSTM + Softmax
def create_LSTM(n_units, input_shape, output_dim, filepath):
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath)

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=output_dim,
                        input_length=input_shape, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(Dropout(0.2))
    model.add(Dense(label_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 通过提供一个输入样本来构建模型
    model.build(input_shape=(None, input_shape))  # 明确指定输入的形状
    #plot_model(model, to_file='./model_lstm.png', show_shapes=True)  # 绘制模型图
    model.summary()  # 输出模型结构

    return model


# 模型训练
def model_train(input_shape, filepath, model_save_path):
    # 将数据集分为训练集和测试集，占比为9：1
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath, input_shape)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=42)

    # 模型输入参数
    n_units = 100
    batch_size = 32
    epochs = 5
    output_dim = 20

    # 模型训练
    lstm_model = create_LSTM(n_units, input_shape, output_dim, filepath)
    lstm_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    # 模型保存
    lstm_model.save(model_save_path)

    # 测试条数
    N = test_x.shape[0]
    predict = []
    label = []
    for start, end in zip(range(0, N, 1), range(1, N + 1, 1)):
        print(f'start:{start}, end:{end}')
        sentence = [inverse_word_dictionary[i] for i in test_x[start] if i != 0]
        y_predict = lstm_model.predict(test_x[start:end])
        print('y_predict:', y_predict)
        label_predict = output_dictionary[np.argmax(y_predict[0])]
        label_true = output_dictionary[np.argmax(test_y[start:end])]
        print(f'label_predict:{label_predict}, label_true:{label_true}')
        # 输出预测结果
        print(''.join(sentence), label_true, label_predict)
        predict.append(label_predict)
        label.append(label_true)

    # 预测准确率
    acc = accuracy_score(predict, label)
    print('模型在测试集上的准确率:%s' % acc)


if __name__ == '__main__':
    filepath = 'all.csv'
    input_shape = 180
    model_save_path = 'corpus_model.h5'
    model_train(input_shape, filepath, model_save_path)
