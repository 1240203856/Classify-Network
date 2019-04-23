# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:10:55 2019

@author: yao
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:40:27 2019

@author: yao
"""
import torch.nn as nn
import argparse
import os
import torch
import torchvision
from torchvision import transforms
from torch import optim
import time
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import f1_score
import torch.nn.functional as F
from collections import OrderedDict
# =============================================================================
# 设置超参数、路径、读取及转换数据
# =============================================================================
#Decice configuration
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser=argparse.ArgumentParser('''Image Classification''')
parser.add_argument('--train_data_path',type=str,default='./data/train_data/',
                    help='''train_image dir path ''')
parser.add_argument('--test_data_path',type=str,default='./data/test_data/',
                    help='''test_image dir path ''')
parser.add_argument('--val_data_path',type=str,default='./data/val_data/',
                    help='''val_image dir path ''')
parser.add_argument('--model_path',type=str,default='./model/',
                    help='''Save model path''')
parser.add_argument('--model_name',type=str,default='densenet121',
                    help='''Model name''')
parser.add_argument('--result_path',type=str,default='./result.txt',
                    help='''Result path''')
parser.add_argument('--label_path',type=str,default='./label.txt',
                    help='''Label path''')
parser.add_argument('--prob_path',type=str,default='./prob.txt',
                    help='''Prob path''')
parser.add_argument('--train',type=bool,default=True,
                    help='Training or not: True/False')
parser.add_argument('--test',type=bool,default=True,
                    help='Testing or not: True/False')
parser.add_argument('--validation',type=bool,default=True,
                    help='Validation or not: True/False')
parser.add_argument('--evaluation',type=bool,default=True,
                    help='Evaluation or not: True/False')
parser.add_argument('--epoch',type=int,default=1,
                    help='''Epochs''')
parser.add_argument('--batch_size',type=int,default=4,
                    help='''batch_size default:50 ''')
parser.add_argument('--lr',type=float,default=0.001,
                    help='Learning rate')
parser.add_argument('--num_classes',type=int,default=3,
                    help='''Num_classes''')
parser.add_argument('--display_epoch',type=int,default=1)
args=parser.parse_args()

# train_datasets dict
item={'Exclude':0,'Negative':1,'Positive':2}    #验证过程中用

if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform1=transforms.Compose([
        transforms.Resize(256), #将图像转化为256*256
        transforms.RandomCrop(224), #从图像中随机裁剪一个224*224的块
        transforms.RandomHorizontalFlip(0.7), #图像有0.7的几率随机旋转
        transforms.ToTensor(), #将numpy数据类型转化为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #图像归一化
    ])

train_data = torchvision.datasets.ImageFolder(root=args.train_data_path,
                                            transform=transform1)
train_loader=torch.utils.data.DataLoader(train_data,batch_size=args.batch_size,
                                           shuffle=True)
test_data = torchvision.datasets.ImageFolder(root=args.test_data_path,
                                            transform=transform1)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=args.batch_size,
                                           shuffle=True)
val_data = torchvision.datasets.ImageFolder(root=args.val_data_path,
                                            transform=transform1)
val_loader=torch.utils.data.DataLoader(val_data,batch_size=args.batch_size,
                                           shuffle=True)

# =============================================================================
# 定义网络结构
# =============================================================================

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=args.batch_size, drop_rate=0, num_classes=args.num_classes):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

def densenet121(pretrained=False):
    model = DenseNet(num_init_features=64,growth_rate=32,block_config=(6,12,24,16))
    if(pretrained==True):
        model = torchvision.models.densenet121(pretrained=True).to(device)
        model.classifier=nn.Linear(1024,args.num_classes)
    return model

def densenet169(pretrained=False):
    model = DenseNet(num_init_features=64,growth_rate=32,block_config=(6,12,32,32))
    if(pretrained==True):
        model = torchvision.models.densenet121(pretrained=True).to(device)
        model.classifier=nn.Linear(1664,args.num_classes)
    return model

def densenet201(pretrained=False):
    model = DenseNet(num_init_features=64,growth_rate=32,block_config=(6,12,48,32))
    if(pretrained==True):
        model = torchvision.models.densenet121(pretrained=True).to(device)
        model.classifier=nn.Linear(1920,args.num_classes)
    return model

def densenet264(pretrained=False):
    model = DenseNet(num_init_features=64,growth_rate=32,block_config=(6,12,64,48))
    if(pretrained==True):
        model = torchvision.models.densenet121(pretrained=True).to(device)
        model.classifier=nn.Linear(2688,args.num_classes)
    return model

model=densenet121(pretrained=False)
# =============================================================================
# 训练
# =============================================================================
if(args.train==True):
    print('Train numbers:',len(train_data))
    cast=nn.CrossEntropyLoss()  #使用交叉熵损失函数
    optimizer=optim.Adam(model.parameters(),lr=args.lr)
    for epoch in range(1,args.epoch+1):
        model.train()
        #start time
        start=time.time()           
        
        for i,data in enumerate(train_loader,0):
            images,labels=data
            images,labels=Variable(images),Variable(labels)
            #Forward pass
            outputs=model(images)
            loss=cast(outputs,labels)
            
            #Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
        if epoch % args.display_epoch==0:
            end=time.time()
            
            print(f"Epoch [{epoch}/{args.epoch}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Time: {(end-start) * args.display_epoch:.1f}sec!")
            
            model.eval()
                
    #Save the model checkpoint
    torch.save(model,args.model_path+args.model_name)
    print("Model save to ",{args.model_path + args.model_name})
         
    print("Finished training!")
# =============================================================================
# 测试
# =============================================================================
if(args.test==True):
    print('Test numbers:',len(test_data))    
    #Load model
    model=torch.load(args.model_path+args.model_name).to(device)
    correct_prediction=0
    total=0
    
    label_list=[]   #将测试集的标签存于一个数组中
    pred=[]         #将测试集的预测值存于一个数组中
    prob0=[]        #将测试集的预测概率存于一个数组中
        
    for i,data in enumerate(test_loader,0):
        # get the inputs
        images,labels=data
        # wrap them in Variable
        images,labels=Variable(images),Variable(labels)    
        outputs=model(images)
        _,predicted = torch.max(outputs.data, 1)    
        total += labels.size(0)
        correct_prediction += (predicted==labels).sum().item()
        #sfmax1为softmax后的概率值
        sfmax1=torch.nn.functional.softmax(outputs.data,dim=1)
        #将概率值由张量转换为数组再转换为列表
        sfmax1=sfmax1.numpy().tolist()
        #将多个列表中的值合并入一个列表
        prob0.extend(sfmax1)
        #将标签值由张量转换为数组
        labels=labels.numpy()
        #将数组中的值按列合并
        label_list=np.concatenate((label_list,labels),0)    # 0为按列合并，1为按行合并
        #将预测值由张量转换为数组
        predicted=predicted.numpy()
        #将数组中的值按列合并
        pred=np.concatenate((pred,predicted),0)    # 0为按列合并，1为按行合并
        
    
    #将概率与标签由列表转换为数组,用其绘制ROC曲线，计算AUC
    prob0=np.array(prob0)                   
    label_list=np.array(label_list)
    
    #将label与predicted写入txt文件中
    file=open(args.prob_path,'w')
    file.write(str(prob0))
    file.close()
    file=open(args.label_path,'w')
    file.write(str(label_list))
    file.close()
    file=open(args.result_path,'w')
    file.write(str(pred))
    file.close()
            
    print(f"Acc: {(correct_prediction / total):4f}")  
         
    print("Finished testing!")

# =============================================================================
# 验证
# =============================================================================
if(args.validation==True):
    print('Val numbers:',len(val_data))
    #Load model
    model=torch.load(args.model_path+args.model_name).to(device)
        
    for i, (images,_) in enumerate(val_loader):
        images=images.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs.data,1)
        di={v:k for k,v in item.items()}
        pred1=di[int(predicted[0])]
        file=str(val_data.imgs[i])[2:-5]
        print(f"{i+1}.({file}) is {pred1}!")
            
    print("Finished validation!")
# =============================================================================
# 评估指标
# =============================================================================
if(args.evaluation==True):
# =============================================================================
# 二分类情况下的Accuracy、Precision、Recall、F1-score
# =============================================================================
    if(args.num_classes==2):
        TP=0
        FP=0
        TN=0
        FN=0   
        for i in range(0,len(label_list)):
            if(label_list[i]==1 and pred[i]==1):
                TP+=1
            elif(label_list[i]==1 and pred[i]==0):
                FN+=1
            elif(label_list[i]==0 and pred[i]==1):
                FP+=1
            elif(label_list[i]==0 and pred[i]==0):
                TN+=1
        
        if((TP+FP)!=0):
            Precision=float(TP)/float(TP+FP)
        else:
            Precision=0
        if((TP+FN)!=0):
            Recall=float(TP)/float(TP+FN)
        else:
            Recall=0        
        Accuracy=float(TP+TN)/float(TN+TP+FN+FP)           
        F1=2*(Precision*Recall)/(Precision+Recall)
        print('Accuracy:',Accuracy)
        print('Precision:',Precision)
        print('Recall:',Recall)
        print('F1-score:',F1)
# =============================================================================
# 二分类情况下的ROC曲线及AUC
# =============================================================================
    if(args.num_classes==2):
        y_test=label_list
        y_score=prob0[:,1]
        print(y_test)
        print(y_score)
        
        fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
        roc_auc = auc(fpr,tpr) ###计算auc的值
        plt.figure()
        lw = 2
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
# =============================================================================
# 多分类情况下的Accuracy、Precision、Recall、F1-score
# =============================================================================
    TP1=0
    FP1=0
    FN1=0
    if(args.num_classes>2):
        for m in range(0,args.num_classes):
            TP=0
            FP=0
            FN=0
            for i in range(0,len(label_list)):   
                if(label_list[i]==m and pred[i]==m):
                    TP+=1
                    TP1+=1
                if(pred[i]==m and label_list[i]!=m):
                    FP+=1
                    FP1+=1
                if(label_list[i]==m and pred[i]!=m):
                    FN+=1
                    FN1+=1
            if((TP+FP)!=0 and(TP+FN)!=0):
                Precision=float(TP)/float(TP+FP)
                Recall=float(TP)/float(TP+FN)
            else:
                Precision=0
                Recall=0
            print('Precision of class',m,':',Precision)
            print('Recall of class',m,':',Recall)
        Accuracy1=float(TP1)/float(total)
        if((TP1+FP1)!=0 and(TP1+FN1)!=0):
            Precision1=float(TP1)/float(TP1+FP1)
            Recall1=float(TP1)/float(TP1+FN1)
        else:
            Precision1=0
            Recall1=0
        print('Accuracy:',Accuracy1)
        print('Precision:',Precision1)
        print('Recall:',Recall1)
        print('F1-score(micro):',f1_score(label_list,pred,average='micro'))
        print('F1-score(macro):',f1_score(label_list,pred,average='macro'))
# =============================================================================
# 多分类情况下的ROC曲线及AUC
# =============================================================================
    if(args.num_classes>2):       
        #将标签二值化
        y_test = label_binarize(label_list, classes=[0, 1, 2])
        y_score=prob0
        # 设置种类
        n_classes = y_test.shape[1]
        # 计算每一类的ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area（方法二）
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Compute macro-average ROC curve and ROC area（方法一）
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot all ROC curves
        lw=2
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
    
    
        





'''
class DenseLayer(nn.Sequential):
    def __init__(self,num_input_features,growth_rate,bn_size,drop_rate):
        super(DenseLayer,self).__init__()
        self.add_module('norm1',nn.BatchNorm2d(num_input_features))
        self.add_module('relu1',nn.ReLU(inplace=True))
        self.add_module('conv1',nn.Conv2d(num_input_features,bn_size*growth_rate,
                                          kernel_size=1,stride=1,bias=False))
        self.add_module('norm2',nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module('relu2',nn.ReLU(inplace=True))
        self.add_module('conv2',nn.Conv2d(bn_size*growth_rate,growth_rate,
                                          kernel_size=3,stride=1,bias=False))
        self.drop_rate=drop_rate
    def forward(self,x):
        new_features=super(DenseLayer,self).forward(x)
        if self.drop_rate>0:
            new_features=F.dropout(new_features,p=self.drop_rate,training=self.training)
            return torch.cat([x,new_features],1)
class DenseBlock(nn.Sequential):
    def __init__(self,num_layers,num_input_features,bn_size,growth_rate,drop_rate):
        super(DenseBlock,self).__init__()
        for i in range(num_layers):
            layer=DenseLayer(num_input_features+i*growth_rate,growth_rate,bn_size,drop_rate)
            self.add_module('denselayer%d'%(i+1),layer)
class Transition(nn.Sequential):
    def __init__(self,num_input_features,num_output_features):
        super(Transition,self).__init__()
        self.add_module('norm',nn.BatchNorm2d(num_input_features))
        self.add_module('relu',nn.ReLU(inplace=True))
        self.add_module('conv',nn.Conv2d(num_input_features,num_output_features,
                                         kernel_size=1,stride=1,bias=False))
        self.add_module('pool',nn.AvgPool2d(kernel_size=2,stride=2))

num_features=0

class DenseNet(nn.Module):
    def __init__(self,growth_rate,block_config,num_input_features,
                 bn_size=args.batch_size,drop_rate=0,num_classes=args.num_classes):
        super(DenseNet,self).__init__()
        
        #First Convolution
        self.features=nn.Sequential(OrderedDict([
               ('conv0',nn.Conv2d(3,num_input_features,kernel_size=7,stride=2,padding=3,bias=False)),
               ('norm0',nn.BatchNorm2d(num_input_features))
               ('relu0',nn.ReLU(inplace=True))
               ('pool0',nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
                ]))
        #Each Denseblock
        num_features=num_input_features
        for i,num_layers in enumerate(block_config):
            block=DenseBlock(num_layers=num_layers,num_input_features=num_features,bn_size=bn_size,
                             growth_rate=growth_rate,drop_rate=drop_rate)
            self.features.add_module('denseblock%d'%(i+1),block)
            num_features=num_features+num_layers*growth_rate
            if i != len(block_config)-1:
                trans=Transition(num_input_features=num_features,num_output_features=num_features//2)
                self.features.add_module('transition%d'%(i+1),trans)
                num_features=num_features//2
        #Final batch norm
        self.features.add_module('norm5',nn.BatchNorm2d(num_features))
        
        #Linear layer
        self.classifier=nn.Linear(num_features,num_classes)
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.constant_(m.bias,0)
    def forward(self,x):
        features=self.features(x)
        out=F.relu(features,inplace=True)
        out=F.adaptive_avg_pool2d(out,(1,1)).view(features.size(0),-1)
        out=self.classifier(out)
        return out

def densenet121(pretrained=False):
    model = DenseNet(num_input_features=64,growth_rate=32,block_config=(6,12,24,16), **kwargs)
    if(pretrained==True):
        model = torchvision.models.densenet121(pretrained=True).to(device)
        model.classifier=nn.Linear(num_features,args.num_classes)
    return model

def densenet169(pretrained=False):
    if(pretrained==False):
        model = DenseNet(num_input_features=64,growth_rate=32,block_config=(6,12,32,32), **kwargs)
    if(pretrained==True):
        model = torchvision.models.densenet121(pretrained=True).to(device)
#        model.fc = nn.Linear(512, args.num_classes)
    return model

def densenet201(pretrained=False):
    if(pretrained==False):
        model = DenseNet(num_input_features=64,growth_rate=32,block_config=(6,12,48,32), **kwargs)
    if(pretrained==True):
        model = torchvision.models.densenet121(pretrained=True).to(device)
#        model.fc = nn.Linear(512, args.num_classes)
    return model

def densenet264(pretrained=False):
    if(pretrained==False):
        model = DenseNet(num_input_features=64,growth_rate=32,block_config=(6,12,64,48), **kwargs)
    if(pretrained==True):
        model = torchvision.models.densenet121(pretrained=True).to(device)
#        model.fc = nn.Linear(512, args.num_classes)
    return model

model=densenet121(pretrained=True)
        
    '''        