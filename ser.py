#!/usr/bin/env python
# coding: utf-8

# # Speech Emotion Recognition

# # Importing the required libraries

# In[1]:


#python
import os

#package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import IPython.display as ipd
import librosa.display
import scipy.io.wavfile

#keras
from keras.utils import np_utils

#sklearn
from sklearn.preprocessing import LabelEncoder

from scipy import signal
from tqdm import tqdm_notebook as tqdm


# # Importing dataset 

# In[2]:

class speech(object):
    dir_list=os.listdir('E:/SER/audio/')
    dir_list.sort()
   # print(dir_list)


# # Pre-processing

# In[3]:


#create Dataframe
    ravdess_db=pd.DataFrame(columns=['path','source','actor','gender','emotion','emotion_lb'])
    count=0
    for i in dir_list:
        file_list=os.listdir('E:/SER/audio/'+i)
        for f in file_list:
            nm=f.split('.')[0].split('-')
            path='E:/SER/audio/'+i+'/'+f
            actor=int(nm[-1])
            emotion=int(nm[2])
            source="Ravdess"
            
            if int(actor)%2==0:
                gender="female"
            else:
                gender="male"
            
            if nm[3]=='01':
                intensity=0
            else:
                intensity=1

            if nm[4]=='01':
                statement=0
            else:
                statement=1

            if nm[5]=='01':
                repeat=0
            else:
                repeat=1
            
            if emotion==1:
                lb="neutral"
            elif emotion==2:
                lb="calm"
            elif emotion==3:
                lb="happy"
            elif emotion==4:
                lb="sad"
            elif emotion==5:
                lb="angry"
            elif emotion==6:
                lb="fearful"
            elif emotion==7:
                lb="disgust"
            elif emotion==8:
                lb="surprised"
            else:
                lb="none"

            ravdess_db.loc[count]=[path,source,actor,gender,emotion,lb]
            count+=1


# In[4]:


   # print(len(ravdess_db))


# In[5]:


    ravdess_db.sort_values(by='path',inplace=True)
    ravdess_db.index=range(len(ravdess_db.index))
    ravdess_db.head()


    # In[6]:


    ravdess_db.to_csv('C:/Users/ayush/Speech Emotion/csv/list.csv')


    # In[7]:


    ravdess_db['split']=np.where((ravdess_db.actor==23) | (ravdess_db.actor==24), 'Test',
                                  (np.where((ravdess_db.actor==21) | (ravdess_db.actor==22),'Val', 'Train')))


    # In[8]:


    ravdess_db['split'].value_counts()


    # In[9]:


    ravdess_db.shape


# In[10]:


    ravdess_db.emotion_lb.value_counts()


    # ### Changing Calm to Neutral

    # In[11]:


    ravdess_db.loc[ravdess_db.emotion_lb=='calm',['emotion','emotion_lb']]=1,'neutral'


    # In[12]:


    ravdess_db.emotion_lb.value_counts()


    # In[13]:


    dataset_db=ravdess_db


    # In[14]:


    dataset_db.emotion_lb=dataset_db.gender+"_"+dataset_db.emotion_lb


# In[15]:


    dataset_db.to_csv('C:/Users/ayush/Speech Emotion/csv/list2.csv')


    # In[16]:


    dataset_db.head()


    # In[17]:


    dataset_db.emotion_lb.value_counts()


    # # Plotting the audio file's waveform

    # In[18]:


    sampling_rate = 44100


    # In[19]:


    filename = ravdess_db.path[0]
    print (filename)


# In[20]:


    samples, sample_rate = librosa.load(filename, res_type='kaiser_fast',sr=sampling_rate)
    sample_rate, samples.shape


    # In[21]:


    ipd.Audio(samples,rate=sample_rate)


    # In[22]:


    # Plotting Wave Form
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Waveform of the audio')
    ax1.set_ylabel('Amplitude')
  #  librosa.display.waveplot(samples, sr=sample_rate)


# ### Trim the Audio
# 

# In[23]:


    samples_trim, index = librosa.effects.trim(samples,top_db=25)
    samples_trim.shape, index


    # In[24]:


    ipd.Audio(samples_trim,rate=sample_rate)


    # In[25]:


    Difference_in_length = len(samples)-len(samples_trim)
    Difference_in_length


    # In[26]:


    Difference_in_duration = librosa.get_duration(samples)-librosa.get_duration(samples_trim)
    Difference_in_duration


# In[27]:


    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Waveform of the Trimmed audio')
    ax1.set_ylabel('Amplitude')
  #  librosa.display.waveplot(samples_trim, sr=sample_rate)


# 
# 

# #### Wiener Filter to remove Noise
# 
# 

# In[28]:


    sample_weiner = scipy.signal.wiener(samples_trim)
    len(sample_weiner)


    # In[29]:


    ipd.Audio(sample_weiner,rate=sample_rate)


    # In[30]:


    Diff_noise = sample_weiner-samples_trim
    ipd.Audio(Diff_noise,rate=sample_rate)


    # In[31]:


    # Plotting Wave Form 
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Waveform of the Trimmed & Filtered audio')
    ax1.set_ylabel('Amplitude')
   # librosa.display.waveplot(sample_weiner, sr=sample_rate)


# #### Waveform of the noise in the audio

# In[32]:


    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Waveform of the Noise')
    ax1.set_ylabel('Amplitude')
  #  librosa.display.waveplot(Diff_noise, sr=sample_rate)


    # In[33]:


    dataset_db.index = range(len(dataset_db.index))


    # In[34]:


    dataset_db.shape


    # # Feature Extraction

    # In[35]:


    audio_duration=3
    sampling_rate=44100
    input_length=sampling_rate * audio_duration
    n_mfcc = 20


# In[36]:


    data_sample = np.zeros(input_length)
    MFCC = librosa.feature.mfcc(data_sample, sr=sampling_rate, n_mfcc=n_mfcc)


    # In[37]:


    MFCC.shape


    # In[38]:


    dataset_db.split.value_counts()


    # In[39]:


    signal, sample_rate = librosa.load(dataset_db.path[0], res_type='kaiser_fast',sr=sampling_rate)
    signal,index = librosa.effects.trim(signal,top_db = 25)
    signal = scipy.signal.wiener(signal)

    if len(signal) > input_length:
        signal = signal[0:input_length]
    elif  input_length > len(signal):
        max_offset = input_length - len(signal)  
        signal = np.pad(signal, (0, max_offset), "constant")


# In[40]:


    signal = np.array(signal).reshape(-1,1)


    # In[41]:


    signal.shape


    # In[42]:


    audios= np.empty(shape=(dataset_db.shape[0],128, MFCC.shape[1], 1))

    count=0
    for i in tqdm(range(len(dataset_db))):
        signal, sample_rate = librosa.load(dataset_db.path[i], res_type='kaiser_fast',sr=sampling_rate)
        signal,index = librosa.effects.trim(signal,top_db = 25)
        signal = scipy.signal.wiener(signal)

        if len(signal) > input_length:
            signal = signal[0:input_length]
        elif  input_length > len(signal):
            max_offset = input_length - len(signal)  
            signal = np.pad(signal, (0, max_offset), "constant")

        melspec = librosa.feature.melspectrogram(signal, sr=sample_rate, n_mels=128,n_fft=2048,hop_length=512)   
        logspec = librosa.amplitude_to_db(melspec)
        logspec = np.expand_dims(logspec, axis=-1)
        audios[count,] = logspec 
        count+=1


# In[43]:


    audios.shape


    # In[44]:


    import h5py
    with h5py.File('Ravdess_audio_Mel_spec.h5', 'w') as hf:
        hf.create_dataset("Ravdess_audio_Mel_spec",  data=audios)


    # In[45]:


    import h5py
    with h5py.File('Ravdess_audio_Mel_spec.h5', 'r') as hf:
      audios = hf['Ravdess_audio_Mel_spec'][:]


    # ### Plotting Mel Power Spectrogram

    # In[46]:


    S_sample = librosa.feature.melspectrogram(sample_weiner, sr=sample_rate, n_mels=128,n_fft=2048,hop_length=512)

    log_S_sample = librosa.amplitude_to_db(S_sample, ref=np.max)

    #plt.figure(figsize=(12, 4))
    #librosa.display.specshow(log_S_sample, sr=sample_rate, x_axis='time', y_axis='mel')
    #plt.title('Mel power spectrogram ')
    #plt.colorbar(format='%+02.0f dB')
    #plt.tight_layout()


    # ### Mel-frequency cepstral coefficients(MFCCs)

    # In[47]:


    mfccs = librosa.feature.mfcc(sample_weiner, sr=sample_rate)

    #plt.figure(figsize=(12, 4))
    #librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
   # plt.ylabel('MFCC coeffs')
   # plt.xlabel('Time')
    #plt.title('MFCC of Audio')
    #plt.colorbar()
    #plt.tight_layout()


# # Dividing The data into train and test

# In[48]:


    x_train=audios[(dataset_db['split']=='Train')]
    y_train=dataset_db.emotion_lb[(dataset_db['split']=='Train')]

   # print(x_train.shape,y_train.shape)


    # In[49]:


    x_test=audios[(dataset_db['split']=='Val')]
    y_test=dataset_db.emotion_lb[(dataset_db['split']=='Val')]

    #print(x_test.shape,y_test.shape)


    # In[50]:


    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_test=np.array(x_test)
    y_test=np.array(y_test)


# In[51]:


    lb=LabelEncoder()
    y_train=np_utils.to_categorical(lb.fit_transform(y_train))
    y_test=np_utils.to_categorical(lb.fit_transform(y_test))


    # In[52]:


    x_traincnn=x_train
    x_testcnn=x_test


    # In[53]:


    x_traincnn.shape,x_testcnn.shape,y_train.shape,y_test.shape


# # Building the model

# In[54]:


    from keras.models import Sequential
    from keras.layers import Conv1D,Conv2D
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    from keras.layers import MaxPooling1D,MaxPooling2D
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.layers import Dense


# In[55]:


    num_classes=len(np.unique(np.argmax(y_train,1)))
    input_shape=x_traincnn.shape[1:]
    learning_rate=0.0001
    decay = 1e-6
    momentum=0.9


    # In[56]:


    input_shape


    # In[57]:


    model=Sequential(name='Audio_CNN_2D')

    model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',data_format='channels_last',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(4,4),strides=(4,4)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(4,4),strides=(4,4)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(4,4),strides=(4,4)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units=num_classes,activation='softmax'))


# In[58]:


#Model Compilation
    from keras import optimizers
    opt=optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,amsgrad=False)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['categorical_accuracy'])


    # In[59]:


   # model.summary()


# # Training the model

# In[60]:


#Train Config
#from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
#batch_size = 16
#num_epochs = 100

#lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)
#mcp_save = ModelCheckpoint('Audio_2DCNN_4L.h5', save_best_only=True, monitor='val_categorical_accuracy', mode='max')
#cnnhistory=model.fit(x_traincnn, y_train, batch_size=batch_size, epochs=num_epochs,validation_data=(x_testcnn, y_test), callbacks=[mcp_save, lr_reduce])


# In[61]:


  #  max(cnnhistory.history['val_categorical_accuracy'])


    # In[62]:


    # Plotting the Train Valid Accuracy Graph

    #plt.plot(cnnhistory.history['categorical_accuracy'])
    #plt.plot(cnnhistory.history['val_categorical_accuracy'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.show()


    # In[63]:


    # Plotting the Train Valid Loss Graph

    #plt.plot(cnnhistory.history['loss'])
    #plt.plot(cnnhistory.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.show()


# # Evaluate the model

# In[96]:


#saving the model.json
    import json
    model_json=model.to_json()
    with open("Audio_2DCNN_LogMelModel_4L.json","w") as json_file:
        json_file.write(model_json)


    # In[97]:


    #loading json and creating model
    from keras.models import model_from_json
    json_file=open("Audio_2DCNN_LogMelModel_4L.json","r") 
    loaded_model_json=json_file.read()
    json_file.close()
    loaded_model=model_from_json(loaded_model_json)


    # In[98]:


    from keras.models import load_model
    #Returns a compiled model identical to the previous one
    loaded_model.load_weights('Audio_2DCNN_4L.h5')


# In[99]:


#evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    score=loaded_model.evaluate(x_testcnn,y_test,verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1],score[1]*100))


    # In[100]:


    x_test_data=audios[(dataset_db['split']=='Test')]
    y_test_data=dataset_db.emotion_lb[(dataset_db['split']=='Test')]
    #print(x_test_data.shape,y_test_data.shape)


    # In[101]:


    preds=loaded_model.predict(x_test_data,batch_size=16,verbose=1)
    pred1=preds.argmax(axis=1)
    abc=pred1.astype(int).flatten()
    predictions=(lb.inverse_transform((abc)))


# In[102]:


    pred_df=pd.DataFrame({'predictedvalues':predictions})
    pred_df[:10]


    # In[103]:


    actual_df=pd.DataFrame({'actualvalues':y_test_data})
    actual_df[:10]
    actual_df.index=range(len(actual_df.index))


    # In[104]:


    final_df=pd.concat([actual_df,pred_df],axis=1)
    final_df.head()


    # In[105]:


    import seaborn as sns
    def print_confusion_matrix(confusion_matrix,class_names,figsize=(9,6),fontsize=14):
        df_cm=pd.DataFrame(confusion_matrix,index=class_names,columns=class_names,)
        fig=plt.figure(figsize=figsize)
        try:
            heatmap=sns.heatmap(df_cm,annot=True,fmt="d")
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),rotation=0,ha='right',fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),rotation=45,ha='right',fontsize=fontsize)
       # plt.ylabel('True label')
        #plt.ylabel('Predicted label')


    # In[106]:


    from sklearn.metrics import accuracy_score
    y_true=final_df.actualvalues
    y_pred=final_df.predictedvalues
    #accuracy_score(y_true,y_pred)*100


    # In[107]:


    from sklearn.metrics import f1_score
    #f1_score(y_true,y_pred,average='macro')*100


# In[108]:


    from sklearn.metrics import confusion_matrix
    c=confusion_matrix(y_true,y_pred)
    #c


    # In[109]:


    class_names=sorted(set(final_df.actualvalues))
    #print_confusion_matrix(c, class_names)


# In[ ]:




