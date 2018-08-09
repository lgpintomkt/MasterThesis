import tensorflow as tf
import dnc
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import confusion_matrix

def data_loader_init(path,unstructured=None,static=False):
    if(unstructured==None):
        return data_loader(path,static)
    else:
        return data_loader(path,static,unstructured=unstructured)
            
def data_loader(path,static=False,unstructured=None):
    if(unstructured==None):
        files=os.listdir(path)
        for file in files:
            if(static==False):
                yield np.load(path+file)
            else:
                data = np.load(path+file)
                shape = data.shape
                if(len(shape)>2):
                    yield data[0,:,:]
                else:
                    yield data
    else:
        files=os.listdir(path)
        for file in files:
            structured_data=np.load(path+file)
            shape = structured_data.shape
            if(len(shape)>2):
                structured_data = structured_data[0,:,:]
            unstructured_data=np.load(unstructured+file)
            yield np.concatenate([structured_data,unstructured_data],axis=1)

class RecommenderSystem():    
    def __init__(self, shape, 
                 classifier='softmax', 
                 dynamic_cell='dnc',
                 embunits=60, 
                 feunits=30, 
                 depth=None, 
                 trees=None, 
                 pretrain=False, 
                 categorical_index=None, 
                 vocabulary=None, 
                 pretrain_sample=None, 
                 set_size=None, 
                 mode='dynamic', 
                 data='both', 
                 kernel_output=3000, 
                 iter_init=None, 
                 target_path=None, 
                 feature_path=None, 
                 vocabulary_size=None, 
                 save_path=None, 
                 unstructured_path=None):
        self.graph = tf.Graph()
        self.shape = shape
        self.batch_size = shape[0]
        self.pretrain_sample = pretrain_sample
        self.classifier = classifier
        self.dynamic_cell = dynamic_cell
        self.feunits = feunits
        self.embunits = embunits
        self.categorical_index=categorical_index
        self.set_size = set_size
        self.kerneloutput = kernel_output
        self.iter_init=iter_init
        self.feature_path=feature_path
        self.target_path=target_path
        self.vocabulary_size=vocabulary_size
        self.save_path=save_path
        self.unstructured_path=unstructured_path
        self.mode=mode
        self.pretrain=pretrain
        self.first_pretrain_iter = True
        
        self.size_tensor = tf.Variable(shape[2])
        
        if(classifier=='decision_tree' or classifier=='random_forest'):
            if(classifier=='decision_tree'):
                trees = 1
            self.feunits = trees * ((2 ** depth)-1)
            
        with self.graph.as_default():
            self.session = tf.Session()
            
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            
            if(mode=='static'):
                self.X = tf.placeholder(tf.float32,shape=(None,shape[1]),name="Input")
            else:
                self.X = tf.placeholder(tf.float32,shape=(None,shape[0],shape[1]),name="Input")
            self.Y = tf.placeholder(tf.float32,shape=(None),name="Label")
            self.initial_state = tf.get_variable(
                            'initial_state', 
                            (self.batch_size,self.feunits),
                            dtype=tf.float32)
            
            #*********************
            #** Embedding Layer **
            #*********************
            if(data=='unstructured' or data=='both'):
                if(pretrain):
                    self.embeddingLayer = tf.layers.dense(inputs=self.X,units=embunits,activation=None,kernel_initializer=self._pre_train_embeddings,name="Embedding")
                else:
                    self.embeddingLayer = tf.layers.dense(inputs=self.X,units=embunits,activation=None,kernel_initializer=tf.random_uniform_initializer(),name="Embedding")
            else:
                self.embeddingLayer = self.X

            #*************************************
            #** Static Feature Extraction Layer **
            #*************************************
            self.staticFeatureExtractionLayer1 = tf.nn.dropout(tf.layers.dense(inputs=self.embeddingLayer,units=self.feunits,activation=tf.nn.relu,name="Static_FE1",kernel_initializer=tf.variance_scaling_initializer()),keep_prob=self.keep_prob)
            self.staticFeatureExtractionLayer2 = tf.nn.dropout(tf.layers.dense(inputs=self.staticFeatureExtractionLayer1,units=20,activation=tf.nn.relu,name="Static_FE2",kernel_initializer=tf.variance_scaling_initializer()),keep_prob=self.keep_prob)

            if(mode=='dynamic'):
            #**************************************
            #** Dynamic Feature Extraction Layer **
            #**************************************
                #Differentiable Neural Computer
                if(dynamic_cell=='dnc'):
                    access_config = {
                            "memory_size": self.batch_size,
                            "word_size": self.feunits,
                            "num_reads": 1,
                            "num_writes": 1,
                      }
                    controller_config = {
                            "hidden_size": self.feunits,
                      }
                    clip_value = 1
                    self.dynamicFeatureExtractionCell = dnc.DNC(access_config, controller_config, self.feunits, clip_value)
                    self.initial_state = self.dynamicFeatureExtractionCell.initial_state(self.batch_size)
                    outputs, _ = tf.nn.dynamic_rnn(
                        self.dynamicFeatureExtractionCell,
                        self.staticFeatureExtractionLayer2,
                        time_major=False,
                        initial_state=self.initial_state,
                        dtype=tf.float32,
                        scope="DNC"
                        )
                else:
                    
                #Recurrent Neural Network
                    if(dynamic_cell=='rnn'):
                        self.dynamicFeatureExtractionCell = tf.contrib.rnn.BasicRNNCell(self.feunits)
                        sequence_fe_name="RNN"
                        
                #Long Short Term Memory Network
                    if(dynamic_cell=='lstm'):
                        self.dynamicFeatureExtractionCell = tf.contrib.rnn.BasicLSTMCell(self.feunits)
                        sequence_fe_name="LSTM"
                    outputs, _ = tf.nn.dynamic_rnn(
                            self.dynamicFeatureExtractionCell,
                            self.staticFeatureExtractionLayer2,
                            time_major=False,
                            dtype=tf.float32,
                            scope=sequence_fe_name
                            )
                
                self.dynamicFeatureExtractionLayer = outputs[:,self.batch_size-1,:] #last output of each user
            
            else:
                self.dynamicFeatureExtractionLayer = self.staticFeatureExtractionLayer2
            
            #**************************
            #** Classification Layer **
            #**************************
            #Softmax
            if(classifier=='softmax'):
                self.classificationLayer = tf.layers.dense(inputs=self.dynamicFeatureExtractionLayer,activation=None,units=10,kernel_initializer=tf.contrib.layers.xavier_initializer(),activity_regularizer=tf.contrib.layers.l2_regularizer(0.001))

            #Kernel Softmax
            elif(classifier=='kernel'):
                self.kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
                        input_dim=20, output_dim=self.kerneloutput, stddev=5.0, name='RFFM')
                self.classificationLayer = tf.layers.dense(inputs=self.kernel_mapper.map(self.dynamicFeatureExtractionLayer),activation=None,units=50,kernel_initializer=tf.contrib.layers.xavier_initializer(),activity_regularizer=tf.contrib.layers.l2_regularizer(1e-6), name="KLR")
  
            #**** Loss ****
            cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.Y, logits=self.classificationLayer, pos_weight=175))+tf.losses.get_regularization_loss()

            self.loss_function = cost
            
            self.opt = tf.train.AdamOptimizer().minimize(cost)
            
            self.threshold = tf.constant(0.5)
            
            init = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            self.session.run([init,init_local])
            
            self.outscore = tf.sigmoid(self.classificationLayer)

            self.pred = tf.where(tf.greater(self.outscore,self.threshold), 
                                 tf.ones(tf.shape(self.classificationLayer)), 
                                 tf.zeros(tf.shape(self.classificationLayer)))
            
            tf.summary.FileWriter(save_path,self.graph)

    def train(self,epochs=1,update=10,patience=1):        
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            if(self.pretrain):
                print('Initiating global network training...')
            else:
                print('Initiating training...')
            for epoch in range(0,epochs):
                epoch_loss = []
                if(self.unstructured_path==None):
                    if(self.mode!='static'):
                        self.features=data_loader_init(self.feature_path)
                        self.validation_features = self.iter_init(self.feature_path.replace("train","validation"))
                    else:
                        self.features=data_loader_init(self.feature_path,static=True)
                        self.validation_features = self.iter_init(self.feature_path.replace("train","validation"),static=True)
                        self.target=self.iter_init(self.target_path.replace("target","unstructured_feature"))
                        self.validation_target=self.iter_init(self.target_path.replace("train","validation").replace("target","unstructured_feature"))
                else:
                    self.target=self.iter_init(self.target_path.replace("target","unstructured_feature"))
                    self.features=data_loader_init(self.feature_path,self.unstructured_path)
                    self.validation_features = self.iter_init(self.feature_path.replace("train","validation"),self.unstructured_path.replace("train","validation"))
                    
                for index in range(0,self.set_size):
                    train_op = self.opt
                    try:
                        train_features = next(self.features)
                        
                        if(self.mode!='static' and self.unstructured_path==None):
                            train_target = np.max(train_features[:,:,449:499],axis=1)
                        elif(self.mode!='static' and self.unstructured_path!=None):
                            train_target = train_features[:,449:499]
                        elif(self.mode=='static' and self.unstructured_path!=None):
                            train_target = train_features[:,449:499]
                        else:
                            train_target = next(self.target)[:,0:50]
                            
                        indices = list(range(449,499))
                        
                        if(self.mode=='static' and self.unstructured_path!=None):
                            train_features = np.transpose(np.array([train_features[:,i] for i in range(0,659) if i not in indices]))
                         
                        if(self.mode!='static' and self.unstructured_path==None):
                            train_features = np.array([train_features[:,:,i] for i in range(0,659) if i not in indices])
                            train_features = train_features.reshape((train_features.shape[1],train_features.shape[2],train_features.shape[0]))
                            
                        _, loss, outputpred = self.session.run([train_op,self.loss_function,self.classificationLayer], feed_dict={
                                    self.X:train_features,
                                    self.Y:train_target.astype(float),
                                    self.keep_prob:1,
                                    self.is_training:False
                                    })
                        epoch_loss.append(loss)
                        
                    except StopIteration:
                        break

                print('Epoch '+str(epoch+1))
                print('{{"metric": "Training Loss", "value": {}}}'.format(np.mean(epoch_loss)))

                validation_cost = tf.nn.weighted_cross_entropy_with_logits(targets=self.Y, logits=self.classificationLayer, pos_weight=175)
                validation_loss_function = tf.reduce_mean(validation_cost)+tf.losses.get_regularization_loss()
                
                validation_loss_vector = []
                for validation_set in range(0,20):
                    val_feat = next(self.validation_features)
                    if(self.mode!='static' or self.unstructured_path!=None):
                        if(self.unstructured_path!=None):
                            batch_y = val_feat[:,449:499]
                            val_feat = np.transpose(np.array([val_feat[:,i] for i in range(0,659) if i not in indices]))
                        else:
                            batch_y = np.max(val_feat[:,:,449:499], axis=1)
                            val_feat = np.array([val_feat[:,:,i] for i in range(0,659) if i not in indices])
                            val_feat = val_feat.reshape((val_feat.shape[1],val_feat.shape[2],val_feat.shape[0]))
                    else:
                        batch_y = next(self.validation_target)[:,0:50]
                    validation_loss_vector.append(
                            self.session.run([validation_loss_function], feed_dict={
                                self.X:val_feat,
                                self.Y:batch_y,
                                self.keep_prob:1,
                                self.is_training:False
                                }))
                current_validation_loss = np.mean(validation_loss_vector)
            
                try:
                    validation_loss
                except NameError:
                    validation_loss = current_validation_loss
                if(current_validation_loss <= validation_loss):
                    self.saver.save(self.session,self.save_path)
                    validation_loss = current_validation_loss
                    print('{{"metric": "Validation Loss", "value": {}}}'.format(validation_loss))
                    print(' ')
                else:
                    patience = patience -1
                    if(patience <= 0):
                        print('Early Stopping Condition Met')
                        print('{{"metric": "Validation Loss", "value": {}}}'.format(np.mean(current_validation_loss)))
                        break
                    else:
                        print('Early Stopping Condition Met but continuing')
                        print('{{"metric": "Validation Loss", "value": {}}}'.format(np.mean(current_validation_loss)))
                self.test(static_feature_path+'test/',target_path+'test/',unstructured_feature_path+'test/')
                  
    def test(self,test_path,target_test_path,unstructured_test_path=None,update=1):
        
        with self.graph.as_default():
            print('Initiating test...')
            batch_y = []
            results = []
            scores = []
            
            if(self.unstructured_path==None):
                if(self.mode!='static'):
                    self.features=self.iter_init(test_path)
                else:
                    self.features=self.iter_init(test_path, static=True)
                    self.target=self.iter_init(target_test_path.replace("target","unstructured_feature"))
            else:
                if(self.mode!='static'):
                    self.features=self.iter_init(test_path,unstructured_test_path)
                    self.target=self.iter_init(target_test_path)
                else:
                    self.features=self.iter_init(test_path,unstructured_test_path, static=True)
            for index in range(0,self.set_size):   
                try:
                    batch = next(self.features)
                    if(self.mode!='static' or self.unstructured_path!=None):
                        indices = list(range(449,499))
                        if(self.unstructured_path==None):
                            batch_y_ = np.max(batch[:,:,449:499], axis=1)
                            batch = np.array([batch[:,:,i] for i in range(0,659) if i not in indices])
                            batch = batch.reshape((batch.shape[1],batch.shape[2],batch.shape[0]))
                        else:
                            batch_y_ = batch[:,449:499]
                            batch = np.transpose(np.array([batch[:,i] for i in range(0,659) if i not in indices]))
                    else:
                        batch_y_ = next(self.target)[:,0:50]
                except StopIteration:
                    break

                results_, scores_, threshold_ = self.session.run([
                        self.pred,
                        self.outscore,
                        self.threshold
                        ], feed_dict={
                            self.X:batch,
                            self.Y:batch_y_.astype(float),
                            self.keep_prob:1,
                            self.is_training:False
                            })
                results.extend(results_)
                scores.extend(scores_)
                batch_y.extend(batch_y_)

            batch_y = np.matrix(batch_y)
            results = np.matrix(results)
            scores = np.matrix(scores)

            jaccard_index = jaccard_similarity_score(batch_y, results)
            
            auc=[]
            for col in range(0,9):
                if(np.sum(batch_y[:,col])!=0):
                    auc.append(roc_auc_score(batch_y[:,col].A,scores[:,col].A))
                else:
                    print('None')
            auc = np.nanmean(auc)
            
            f1=[]
            for col in range(0,9):
                f1.append(f1_score(batch_y[:,col], results[:,col]))
            f1 = np.mean(f1)
            
            tp=[]
            tn=[]
            fp=[]
            fn=[]
            for col in range(0,9):
                true_n, false_p, false_n, true_p = confusion_matrix(batch_y[:,col], results[:,col]).ravel()
                tp.append(true_p)
                tn.append(true_n)
                fp.append(false_p)
                fn.append(false_n)
            tp = np.sum(tp)
            tn = np.sum(tn)
            fp = np.sum(fp)
            fn = np.sum(fn)
            
            print(" ")
            print("Jaccard Index: "+str(jaccard_index))
            print("AUC: "+str(auc))
            print("F1 Score: "+str(f1))
            print(" ")
            print("Positives "+str(int(np.sum(results.A1))))
            print("Sample Positives: "+str(int(np.sum(batch_y.A1))))
            print("Examples: "+str(len(batch_y.A1)))
            print(' ')
            print("TP: "+str(tp))
            print("TN: "+str(tn))
            print("FP: "+str(fp))
            print("FN: "+str(fn))
            print(' ')
                
    def _num_embed(self, features, categorical=False, reboot=False):
        
        if(reboot == True):
            self.first_pretrain_iter = True
            return True
        else:
            # Construct model
            with self._autoencoderGraph.as_default():
        
                    if(self.first_pretrain_iter==True):
                        print('Constructing Graph')
                        self.X_pt = tf.placeholder(tf.float32,shape=features.shape)
                        
                        if(categorical==False):
                            self.encoder_op = tf.layers.dense(self.X_pt, units=int(self.embunits/4), activation=None, name='encoder')
                            self.decoder_op = tf.layers.dense(self.encoder_op, units=features.shape[1], activation=tf.nn.softmax)
                        else:
                            self.encoder_op = tf.layers.dense(self.X_pt, units=int((3*self.embunits)/4), activation=None, name='encoder')
                            self.decoder_op = tf.layers.dense(self.encoder_op, units=features.shape[1], activation=None)

                        # Prediction
                        self.y_pred_pt = self.decoder_op
                        
                        # Targets (Labels) are the input data.
                        self.y_true_pt = self.X_pt
                
                        # Define loss and optimizer, minimize the squared error
                        if(categorical==False):
                            self.loss_pt = tf.reduce_mean(tf.pow(self.y_true_pt - self.y_pred_pt, 2))
                        else:
                            self.loss_pt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true_pt, logits=self.y_pred_pt))
                        self.optimizer = tf.train.RMSPropOptimizer(0.01).minimize(self.loss_pt)
                
                        # Initialize the variables (i.e. assign their default value)
                        self.init_pt = tf.global_variables_initializer()
                        print('Training')
                    
                    # Start Training
                    # Start a new TF session
                    with tf.Session() as sess:
            
                        # Run the initializer
                        sess.run(self.init_pt)
            
                        # Training
                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, outputs, scores = sess.run([self.optimizer, self.loss_pt, self.y_pred_pt], feed_dict={self.X_pt: features})
                        
                        with tf.variable_scope("encoder", reuse=True):
                            weights = tf.get_variable("kernel")   
                        weights = sess.run(weights)
                    
                    if(self.first_pretrain_iter == True):
                        self.first_pretrain_iter = False
                     
            return weights, scores
    
    def _MySentences(self,df):
        for row in df.itertuples():
            rowTokens = []
            for col,value in enumerate(list(row[1:])):
                if(value==1):
                    rowTokens.append(str(df.columns.values.tolist()[col]))
            if(len(rowTokens)==0):
                continue
            else:
                yield rowTokens

    def _pre_train_embeddings(self, shape, dtype=tf.float32, partition_info=None):
        print('Initiating Embedding Layer Pretraining')
        vocabulary=[]
        for col in range(self.vocabulary_size):
            vocabulary.append('col'+str(col))
        
        #numerical
        if(self.mode == 'static'):
            pretrain_iterator = data_loader(path = static_feature_path+'train/',unstructured = unstructured_feature_path+'train/', static = True)
        else:
            pretrain_iterator = data_loader(path = static_feature_path+'train/',unstructured = unstructured_feature_path+'train/', static = True)      
        
        print('Initiating Pre-training for Numeric Variables')
        self._autoencoderGraph = tf.Graph()
        features = []
        scores = []
        epoch=1
        while(1):
            try:
                X = pd.DataFrame(next(pretrain_iterator))
            except StopIteration:
                if(epoch<2):
                    epoch+=1
                    if(self.mode == 'static'):
                        pretrain_iterator = data_loader(path = static_feature_path+'train/',unstructured = unstructured_feature_path+'train/', static = True)
                    else:
                        pretrain_iterator = data_loader(path = static_feature_path+'train/',unstructured = unstructured_feature_path+'train/', static = True)      
                else:
                    break
            
            rows,columns = X.shape
            
            numericData = X.iloc[:,0:self.categorical_index]
            numRows,numCols = numericData.shape
            numeric_embeddings, results =self._num_embed(numericData)
            features.append(numericData)
            scores.append(results)

        print('Pre-training for Numeric Variables Finished')

        #categorical
        if(self.mode == 'static'):
            pretrain_iterator = data_loader(path = static_feature_path+'train/',unstructured = unstructured_feature_path+'train/', static = True)
        else:
            pretrain_iterator = data_loader(path = static_feature_path+'train/',unstructured = unstructured_feature_path+'train/', static = True)      
        
        self._num_embed(numericData,reboot=True)
        
        print('Initiating Pre-training for Categorical Variables')
        self._autoencoderGraph = tf.Graph()
        epoch=1
        while(1):
            try:
                X = next(pretrain_iterator)
                indices = list(range(449,499))
                X = np.transpose(np.array([X[:,i] for i in range(0,659) if i not in indices]))
                X = pd.DataFrame(X)
            except StopIteration:
                if(epoch<2):
                    epoch+=1
                    if(self.mode == 'static'):
                        pretrain_iterator = data_loader(path = static_feature_path+'train/',unstructured = unstructured_feature_path+'train/', static = True)
                    else:
                        pretrain_iterator = data_loader(path = static_feature_path+'train/',unstructured = unstructured_feature_path+'train/', static = True)      
                else:
                    break
            
            rows,columns = X.shape
            
            ceneData = X.iloc[:,self.categorical_index:columns]
            
            ceneRows,ceneColumns = ceneData.shape
            
            cene_embeddings, scores =self._num_embed(ceneData,categorical=True)
        
        print('Concatenating Embeddings')
        minimum=min([np.amin(numeric_embeddings),np.amin(cene_embeddings)])
        maximum=max([np.amax(numeric_embeddings),np.amax(cene_embeddings)])
        
        catRows,catCols = cene_embeddings.shape
        numRows,numCols = numeric_embeddings.shape

        extraCols = np.random.uniform(low=minimum/100,high=maximum/100,size=(numRows,catCols))
        extraRows = np.random.uniform(low=minimum/100,high=maximum/100,size=(catRows,numCols))

        merged1 = np.concatenate([numeric_embeddings,extraCols],axis=1)
        merged2 = np.concatenate([extraRows,cene_embeddings],axis=1)
        
        return np.concatenate([merged1,merged2],axis=0)

cols=['id','timestamp','churn']
timeseries_size = 30
batch_size = 2000
categoricalIndex = 175-len(cols)

model_save_path='/output/tfmodels/'
unstructured_feature_path='/data/unstructured_feature_batches/'
static_feature_path='/data/static_feature_batches/'
dynamic_feature_path='/data/dynamic_feature_batches/'
target_path='/data/target_batches/'

print('Model 1 - Softmax classifier with structured static features')
_,num_cols=next(data_loader(static_feature_path+'train/', static=True)).shape
    
model1 = RecommenderSystem(
        shape=(timeseries_size,num_cols,batch_size),
        classifier='softmax',
        set_size=batch_size,
        mode='static',
        iter_init=data_loader_init,
        feature_path=static_feature_path+'train/',
        target_path=target_path+'train/',
        save_path=model_save_path+'model1/')

model1.train(epochs=100)
model1.test(static_feature_path+'test/',target_path+'test/')

print(' ')
print('Model 2 - Softmax classifier with unstructured static features')
_,num_cols=next(data_loader(path = static_feature_path+'train/',unstructured = unstructured_feature_path+'train/', static = True)).shape
 
model2 = RecommenderSystem(
        shape=(timeseries_size,num_cols,batch_size),
        classifier='softmax',
        categorical_index=categoricalIndex,
        pretrain=False,
        vocabulary_size=num_cols-categoricalIndex,
        set_size=batch_size,
        mode='static',
        iter_init=data_loader_init,
        feature_path=static_feature_path+'train/',
        unstructured_path=unstructured_feature_path+'train/',
        target_path=target_path+'train/',
        save_path=model_save_path+'model2/')

model2.train(epochs=100, patience=1)
model2.test(static_feature_path+'test/',target_path+'test/',unstructured_feature_path+'test/')

print(' ')
print('Model 3 - Softmax classifier with unstructured static features andlayer pretraining')
_,num_cols=next(data_loader(path = static_feature_path+'train/',unstructured = unstructured_feature_path+'train/', static = True)).shape
 
model3 = RecommenderSystem(
        shape=(timeseries_size,num_cols,batch_size),
        classifier='softmax',
        categorical_index=categoricalIndex,
        pretrain=True,
        vocabulary_size=num_cols-categoricalIndex,
        set_size=batch_size,
        mode='static',
        iter_init=data_loader_init,
        feature_path=static_feature_path+'train/',
        unstructured_path=unstructured_feature_path+'train/',
        target_path=target_path+'train/',
        save_path=model_save_path+'model2/')

model2.train(epochs=100, patience=1)
model2.test(static_feature_path+'test/',target_path+'test/',unstructured_feature_path+'test/')

print(' ')
print('Model 4 - Kernel Logistic Regression classifier with unstructured static features')
_,num_cols=next(data_loader(path = static_feature_path+'train/',unstructured = unstructured_feature_path+'train/', static = True)).shape
 
model4 = RecommenderSystem(
        shape=(timeseries_size,num_cols,batch_size),
        classifier='kernel',
        categorical_index=categoricalIndex,
        pretrain=True,
        vocabulary_size=num_cols-categoricalIndex,
        set_size=batch_size,
        mode='static',
        iter_init=data_loader_init,
        feature_path=static_feature_path+'train/',
        unstructured_path=unstructured_feature_path+'train/',
        target_path=target_path+'train/',
        save_path=model_save_path+'model3/')

model4.train(epochs=100, patience=1)
model4.test(static_feature_path+'test/',target_path+'test/',unstructured_feature_path+'test/')