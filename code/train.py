# coding=utf-8
import ctypes
import glob

import time
import os

from tqdm import tqdm
from PIL import Image



import LiteMRINet





from seg_utils import data_loader

from seg_utils.loss import *

import numpy as np


import torchvision
import config

import 绘制曲线图





import warnings
# 忽略包含特定文本的UserWarning
warnings.filterwarnings('ignore', category=UserWarning, message='Some warning text')


batch_size =config.batch_size
n_epoch = config.n_epoch
img_size = config.img_size
#model_name = 'CLNet'

dirPath=config.dirPath
modeName=config.modeName

model_path = dirPath+modeName+"/"
predictPath=model_path+"predictResult/"

op_lr =config.op_lr       #学习率
weightDecay=config.weightDecay      # L2正则化权重
#op_decay_rate = 0.1     #学习率衰减比例
#op_decay_epoch = 60      #学习率衰减间隔
noChangeEpoch=config.noChangeEpoch       #终止条件


if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(predictPath):
    os.makedirs(predictPath)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


net=LiteMRINet.LiteMRINet(3,2).to(device)





train_loader = data_loader.get_loader(dirPath+"train/" , batch_size , img_size,num_workers=4, mode='train',augmentation_prob=0,shuffle=True, pin_memory=True)
val_loader = data_loader.get_loader(dirPath+"val/" , 1 ,  img_size,num_workers=4, mode='val',augmentation_prob=0,shuffle=False, pin_memory=True)
# test_loader = data_loader.get_loader(dirPath+"test/" , 1 , img_size,num_workers=4, mode='test',augmentation_prob=0,shuffle=False, pin_memory=True)
predict_loader = data_loader.get_loader(dirPath+"predict/" , 1 , img_size,num_workers=4, mode='predict',augmentation_prob=0,shuffle=False, pin_memory=True)


criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(net.parameters(),lr=op_lr,eps=1e-3)
#optimizer =torch.optim.SGD(net.parameters(), lr=op_lr, momentum=0.9,weight_decay=0) 
#scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
epoch=0

def main():
    global epoch
    #看是否是继续训练
    epoch=getLastCheckPt(model_path)#得到应该训练的周期
    if epoch!=1:#有模型，从已有最新的周期恢复训练参数
            
        checkpoint = torch.load(model_path+str(epoch-1)+".pth",map_location={'cuda:5':'cuda:0'})            
        net.load_state_dict(checkpoint)        # 从字典中依次读取
        print("已恢复第{}个周期的参数".format(epoch-1))


    isTrain=False
    isTrain=True
    while isTrain:#循环训练num_epoches个epoch
            
        trainOneEpoch()
            
        evaluateOneEpoch()

        #绘制曲线图.draw(block=False)
        # 绘制曲线图.printMaxMIOU()
        if stop():
            print("训练完成")
            for i in range(10):
                # player = ctypes.windll.kernel32
                # player.Beep(500 * i, 5000)
                print()
            break

    

def stop():
    # 终止条件
    # 读取已有的epoch指标
    allValLoss = []
    allValMiou = []
    with open(model_path + '评价指标.txt', 'a+') as f:  # 设置文件对象
        f.seek(0.0)  # 将指针移到开头
        while True:  # 读取所有的数据
            text = f.readline()  # 只读取一行内容

            # 判断是否读取到内容
            if not text:
                break

            texts = text.split(":")
            if texts[0] == "loss":
                allValLoss.append(float(texts[1]))  # 将loss加到Loss列表中。
            if texts[0] == "mIoU":
                allValMiou.append(float(texts[1]))  # 将loss加到Loss列表中。

    # 删除多余的pth
    bestNum = allValMiou.index(max(allValMiou)) + 1
    lastNum = len(allValMiou)
    for filename in glob.glob(os.path.join(model_path, '*.pth')):
        if (os.path.split(filename)[-1] != '{}.pth'.format(bestNum)) & (
                os.path.split(filename)[-1] != '{}.pth'.format(lastNum)):
            os.remove(filename)
    #################################

    maxIndex = allValMiou.index(max(allValMiou)) + 1


    minIndex = allValLoss.index(min(allValLoss)) + 1
    if ((
                epoch - minIndex) > noChangeEpoch or epoch > n_epoch) and epoch > 100:  # 如果超过10个epoch还没有更高的epoch，或者训练总周期已经超过epochRepeatNum，那么中止训练。
        with open(model_path + '评价指标.txt', 'a') as f:  # 设置文件对象
            f.write("\nbestEpoch=" + str(maxIndex) + "\n")

        return True
    else:
        return False

def getLastCheckPt(model_path):
   
    files=os.listdir(model_path)
    max=0
    for file in files:
        if file[-4:]!=".pth":
            continue

        epoch,Extension =file.split(".")
        epoch=int(epoch)
        
        if max< epoch:
            max=epoch
    return max+1 #新的周期数


def trainOneEpoch():
    

    net.train(True)  
        
    print("\n正在训练第{}个周期".format(epoch))


    # cur_lr = adjust_lr(optimizer, op_lr, epoch, op_decay_rate, op_decay_epoch)#更新学习率
    #print("当前学习率为",cur_lr)

    #epoch_loss = 0
    trainLoss = 0.  
    #miou_score = 0.  
    #f1_score = 0. 
    #rawf1_score=0 
    #accuracy_score = 0.  
    #P=0.
    #R=0.
    eTP=0
    eTN=0
    eFP=0
    eFN=0
    length = 0.
    st = time.time()
    for i,(inputs, mask) in enumerate(tqdm(train_loader)): #每个批
        # print("inputs",inputs.shape)
        X = inputs.to(device)
        # print("X",X.shape)
        Y = mask.to(device).long()




        output = net(X)                                 #推理出结果

        gt=torch.squeeze(Y,1)
        loss=torch.nn.CrossEntropyLoss(reduce=True)(output,gt) #CE算出的是所有样本损失的平均值
        #print(GT_flat)

        #计算L2损失
        L2Loss=0
        for para in net.parameters():
            L2Loss+=torch.sum(para*para)
        loss+=weightDecay*L2Loss
        # loss = losses.mean()
        #loss = criterion(SR_flat, GT_flat)



        optimizer.zero_grad()#清空梯度
        loss.backward()#反向传播

        optimizer.step()#更新模型参数

        ##每个周期更新学习率
        #torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1).step()
        #scheduler.step()
        trainLoss += loss.item()
        st = time.time()
            
        #print(torch.min(SR_eva))
        #SR_eva[SR_eva >= 0.5] = 1
        #SR_eva[SR_eva < 0.5] = 0  
            
        output=F.softmax(output,dim=1)
        SR_eva=torch.argmax(output, dim=1, keepdim=False)

        TN,FP,FN,TP=BCM(SR_eva,Y)
        eTN+=TN
        eFP+=FP
        eFN+=FN 
        eTP+=TP
            
        length += 1
        #print("第{f:}步",length)
        #print("loss:",loss.item())
        #print("miou_score:",getmIoU(SR_eva, Y))
        #print("f1_score:",getF1score(SR_eva, Y))
        #print("accuracy_score:",getACC(SR_eva,Y))

    #得到每个epoch的精度指标
    eP=eTP/(eTP+eFP+1e-7)
    eR=eTP/(eTP+eFN+1e-7)
    iou0=eTN/(eFP+eTN+eFN+1e-7)
    iou1=eTP/(eFP+eTP+eFN+1e-7)

    miou_score=round((iou0+iou1)/2*100,1)
    f1_score=round(2*eP*eR/(eP+eR+1e-7)*100,1)
    accuracy_score=round((eTP+eTN)/(eTP+eFP+eFN+eTN+1e-7)*100,1)

    #accuracy_score = round(accuracy_score / length*100,1)
    #miou_score = round(miou_score / length*100,1)
    #f1_score= round(f1_score  / length*100,1)
    #rawf1_score= round(rawf1_score  / length*100,1)
    trainLoss = round(trainLoss/length,3)

    #P= round(P  / length*100,1)
    #R= round(R  / length*100,1)

            
        

    print(  "\nTrain Miou: %g %%" % (miou_score),
            "\nTrain F1-score: %g %%" % (f1_score),
          
            "\nTrain accuracy: %g %%" % (accuracy_score),
            "\ntrain loss: %g %%" % (trainLoss),
           
            )

######################每个epoch后开始评估=========================================================================================
def evaluateOneEpoch():
    print("Start Evaluating!")
    global epoch
    valLoss = 0.  
    #miou_score = 0.  
    #f1_score = 0.  
    #accuracy_score = 0.  
    #P=0.
    #R=0.
    eTP=0
    eTN=0
    eFP=0
    eFN=0
    length = 0.
    #net.train(False)
    net.eval()
    for i, (inputs, mask) in enumerate(tqdm(val_loader)):
        with torch.no_grad():

            X = inputs.to(device)
        #    Y = mask.to(device)
               
        #    #optimizer.zero_grad()
        #    output = net(X)
        #    SR_probs = SR_eva = F.sigmoid(output)

        #    SR_flat = SR_probs.view(SR_probs.size(0), -1)
        #    GT_flat = Y.view(Y.size(0), -1)
        ## loss = losses.mean()
            Y = mask.to(device).long()



            output = net(X)                                 #推理出结果
                
            gt=torch.squeeze(Y,1)
            loss=torch.nn.CrossEntropyLoss()(output,gt)
                
            L2Loss=0
            for para in net.parameters():
                L2Loss+=torch.sum(para*para)

            loss+=weightDecay*L2Loss


            valLoss +=loss.item()/batch_size

            #SR_eva[SR_eva >= 0.5] = 1
            #SR_eva[SR_eva < 0.5] = 0
            output=F.softmax(output,dim=1)
            SR_eva=torch.argmax(output, dim=1, keepdim=False)

            
            TN,FP,FN,TP=BCM(SR_eva,Y)
            eTN+=TN
            eFP+=FP
            eFN+=FN 
            eTP+=TP
            #print("评估每一步的miou:",miou_score)
            #print("评估每一步的f1_score:",f1_score)

            length += 1

    #得到所有评估样本的指标
    eP=eTP/(eTP+eFP+1e-7)
    eR=eTP/(eTP+eFN+1e-7)
    iou0=eTN/(eFP+eTN+eFN+1e-7)
    iou1=eTP/(eFP+eTP+eFN+1e-7)

    miou_score=round((iou0+iou1+1e-7)/2*100,1)
    f1_score=round(2*eP*eR/(eP+eR+1e-7)*100,1)
    accuracy_score=round((eTP+eTN)/(eTP+eFP+eFN+eTN+1e-7)*100,1)
    # print("length:",length)

    valLoss =  round(valLoss/length,3)
    #accuracy_score =  round(accuracy_score / length*100,1)
    #miou_score =  round(miou_score /length*100,1)
    #f1_score=  round(f1_score  /length*100,1)
    

   
    #写入新的数据
    with open(model_path+'评价指标.txt','a+') as f:    #设置文件对象 
        #写入新的数据
        f.write("\nepoch="+str(epoch)+"\n")
        #f.write("Loss:"+str(val_loss)+"\n")
        f.write("mIoU:"+str(miou_score)+"\n")                 
        #f.write("precision:"+str(precision_score)+"\n") 
        f.write("F1-score:"+str(f1_score)+"\n") 
        f.write("accuracy:"+str(accuracy_score)+"\n") 
        f.write("loss:"+str(valLoss)+"\n")

    #每个周期保存模型
    #state = {'state': net.state_dict(),'epoch': epoch}                   # 将epoch一并保存
    
    torch.save(net.state_dict(), model_path+"/"+str(epoch)+".pth")
    epoch=epoch+1
    #unet_score = JS + DC
    
    print(
            "\nVal Miou: %g %%" % (miou_score),
            "\nVal F1-score: %g %%" % (f1_score),
               
            "\nVal accuracy: %g %%" % (accuracy_score),
            "\nVal loss: %g %%" % (valLoss),
           
            )
    绘制曲线图.printMaxMIOU()
def test():
    print("Start Testing!")
    valLoss = 0.  
    #miou_score = 0.  
    #f1_score = 0.  
    #accuracy_score = 0.  
    #P=0.
    #R=0.
    eTP=0
    eTN=0
    eFP=0
    eFN=0
    length = 0.
    #net.train(False)
    net.eval()
    for i, (inputs, mask) in enumerate(tqdm(test_loader)):
        with torch.no_grad():

            X = inputs.to(device)

            Y = mask.to(device).long()

            output = net(X)                                 #推理出结果
                
            gt=torch.squeeze(Y,1)
            loss=torch.nn.CrossEntropyLoss()(output,gt)
                
            L2Loss=0
            for para in net.parameters():
                L2Loss+=torch.sum(para*para)

            loss+=weightDecay*L2Loss 


            valLoss +=loss.item()

            #SR_eva[SR_eva >= 0.5] = 1
            #SR_eva[SR_eva < 0.5] = 0
            output=F.softmax(output,dim=1)
            SR_eva=torch.argmax(output, dim=1, keepdim=False)

            
            TN,FP,FN,TP=BCM(SR_eva,Y)
            eTN+=TN
            eFP+=FP
            eFN+=FN 
            eTP+=TP
            #print("评估每一步的miou:",miou_score)
            #print("评估每一步的f1_score:",f1_score)

            length += 1

    #得到所有评估样本的指标
    eP=eTP/(eTP+eFP)
    eR=eTP/(eTP+eFN)    
    iou0=eTN/(eFP+eTN+eFN)
    iou1=eTP/(eFP+eTP+eFN)

    miou_score=round((iou0+iou1)/2*100,1)
    f1_score=round(2*eP*eR/(eP+eR)*100,1)
    accuracy_score=round((eTP+eTN)/(eTP+eFP+eFN+eTN)*100,1)

    valLoss =  round(valLoss/length,3)

    
    print(
            "\nVal Miou: %g %%" % (miou_score),
            "\nVal F1-score: %g %%" % (f1_score),
               
            "\nVal accuracy: %g %%" % (accuracy_score),
            "\nVal loss: %g %%" % (valLoss),
           
            )
    
def predict():
    print("\nStart Predicting!")
    net.eval()
    for i, (inputs, filename) in enumerate(tqdm(predict_loader)):#每个批
        X = inputs.to(device)
        #Y = mask.to(device)
        output = net(X)
        # output = net(X)
        output = F.softmax(output)
        for i in range(output.shape[0]):#对每个批的每个图像推理结果分开处理
            #probs_array = (torch.squeeze(output[i])).data.cpu().numpy()
            array=torch.squeeze(output[i])
            mask_array = torch.argmax(array, axis=0).data.cpu().numpy()
            final_mask = mask_array.astype(np.float32)
            final_mask = final_mask * 255
            final_mask = final_mask.astype(np.uint8)
            # print(predictPath + filename[i] + '.png')
            final_savepath = predictPath + filename[i] + '.png'
            im = Image.fromarray(final_mask)
            im.save(final_savepath)
def BCM(pred,gt):#��������
    #ȡ�����ֵ
    #pred=torch.round(pred, out=None)
    #gt=torch.round(gt, out=None)
    TP=float(torch.sum((pred==1)&(gt==1)))
    FP=float(torch.sum((pred==1)&(gt==0)))
    FN=float(torch.sum((pred==0)&(gt==1)))
    TN=float(torch.sum((pred==0)&(gt==0)))
    #print(TP+FP+FN+TN)
    #print(torch.max(gt))
    #print(FN)
    #print(FP)
    return TN,FP,FN,TP
if __name__ == '__main__':

    main()