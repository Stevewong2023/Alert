import torch
import torch.nn as nn
from model import My_model
from dataprocessor import Mydata
from torch.utils.data import DataLoader
import argparse
import os
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from parameter import ValParams
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_event_loss_coefficient(event_typer):
    lcoe = torch.ones_like(event_typer)
    for i, class_ in enumerate(event_typer):
        if int(class_) in [0,2,3]:
            lcoe[i] = 0
    return lcoe
def get_loss_coefficient(typer, sore):
    lcoe = torch.ones_like(typer)
    if sore == 'start':
        for i, class_ in enumerate(typer):
            if class_ ==1 or class_ == 2 or class_ == 3:    
                lcoe[i] = 0
    elif sore == 'end':
        for i, class_ in enumerate(typer):
            if class_ == 0 or class_ ==1 or class_ == 3:    
                lcoe[i] = 0
    return lcoe
def ploting(name,data,start_pre,end_pre,start_label,end_label,type,pre_phase):
    filename = f'{type}'
    os.makedirs(f'画图/训练集512_g-->n/{filename}/',exist_ok=True)
    df=pd.DataFrame(np.array(data.tolist()).squeeze(),columns=['label:'+str(type)+'||'+'prediction:'+str(pre_phase)])
    df.plot(figsize=(16,8),alpha=1,marker=',',grid=False,fontsize='16')
    plt.gca().invert_yaxis()
    plt.axvline(x=int(start_pre.cpu()), color='b',linestyle='-')
    plt.axvline(x=int(end_pre.cpu()), color='b',linestyle='-')
    plt.axvline(x=int(start_label.cpu()), color='r',linestyle='-')
    plt.axvline(x=int(end_label.cpu()), color='r',linestyle='-')
    # plt.yticks([5, 10])
    #plt.show()
    plt.savefig(f'画图/训练集512_g-->n/{filename}/'+f'/{str(name)}.png')
    plt.close()
if __name__ == '__main__':
    # root = '/raid/wxt/wxt/短时标稀有天体事件/'
    parse = argparse.ArgumentParser()
    parse.add_argument('--traindir',type=list,default=['train_dataset/p4-cpg_vv4_vv3_3f_1024_0.15_0.35_dataset'])
    parse.add_argument('--labeldir',type=list,default=['train_dataset/p4-cpg_vv4_vv3_3f_1024_0.15_0.35_label'])
    # parse.add_argument('--traindir',type=list,default=['valid_dataset/p4v_cpg_512_0.2to0.5_data'])
    # parse.add_argument('--labeldir',type=list,default=['valid_dataset/p4v_cpg_512_0.2to0.5_label'])
    parse.add_argument('--validdir',type=list,default=['valid_dataset/p4v_cpg_512_0.2to0.5_data'])
    parse.add_argument('--vlabeldir',type=list,default=['valid_dataset/p4v_cpg_512_0.2to0.5_label'])

    
    parse.add_argument('--window',type=int,default=1024)
    parse.add_argument('--mode',type=str,default='trai')
    parse.add_argument('--epochs',type=int,default=1000000)
    parse.add_argument('--batchsize',type=int,default=512)
    parse.add_argument('--lr',type=float,default=1e-5)
    parse.add_argument('--distance',type=int,default=50)
    parse.add_argument('--root',type=str,default='AlertE_cancel_coding/')
    parse.add_argument('--weight',type=str,default='AlertE_cancel_coding/weight/8_0.pth')
    # parse.add_argument('--weight',type=str,default='AlertE_just4micro/weight/16_3058.pth')
    args = parse.parse_args(args=[])

    '''
    device_id=0
    torch.cuda.set_device(device_id)
    device = 'cuda:0' if torch.cuda.is_available else 'cpu'
    '''
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    activate_fn=nn.Sigmoid()
    # Model=E_Decoder(deepth_D=8,dim_D=83,dim_E=45,heads_E=8,heads_D=8,window=1024,deepth_E=16,device=device,batch=args.batchsize)
    Model = My_model(dim = 128, depth = 8, heads = 16, mlp_dim = 1024, 
                    window = 512, device = device, kernels = 64, 
                    kernel_size = 4, stride = 4, dim_head = 192, dropout = 0.1)
    Model = nn.DataParallel(Model, device_ids=[0]).to(device)
    Model = Model.to(device)
    vdataset=Mydata(dataset_dir=args.validdir,label_dir=args.vlabeldir,window=args.window)
    valid_dataloader=DataLoader(dataset=vdataset,batch_size=1,shuffle=True,num_workers=1,drop_last=False)
    optimizer=torch.optim.Adam(Model.parameters(),lr=args.lr)
    #optimizer=torch.optim.SGD(Model.parameters(),lr=1e-3,momentum=0.9)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, verbose=True)
    cl_CELoss   = nn.CrossEntropyLoss()
    cl_CELoss1 = nn.CrossEntropyLoss(reduction = 'none')
    lum_start_CELoss   = nn.CrossEntropyLoss(reduction = 'none')
    lum_end_CELoss   = nn.CrossEntropyLoss(reduction = 'none')
    project_root=args.root
    writer=SummaryWriter(project_root+'logs'+'/'+str(current_time))
    def validation(weight):
        print('\nvalidating...')
        print(weight)
        Model.eval()
        with torch.no_grad():
            for x in range(1):
                checkpoint=torch.load(weight,map_location=device)
                # Model.load_state_dict(checkpoint['m'])
                if device == 'cpu':
                    # 加载模型参数文件
                    # state_dict = torch.load('your_model.pth')
                    # 创建一个新的 state_dict
                    new_state_dict = OrderedDict()
                    # 去掉 'module.' 前缀
                    for k, v in checkpoint['m'].items():
                        if k.startswith('module.'):
                            k = k[7:]  # 去掉 'module.' 前缀
                        new_state_dict[k] = v

                    # 加载新的 state_dict 到模型
                    Model.load_state_dict(new_state_dict)
                else:
                    Model.load_state_dict(checkpoint['m'])
                params=ValParams()
                for  i,item in tqdm(enumerate(valid_dataloader)):
                    params.total_num+=1             
                    data,target = item
                    starter_v, ender_v, phase_typer_v, event_typer_v= target['start'].to(device), target['end'].to(device), target['phase_type'].to(device), target['event_type'].to(device)
                    data = data.to(device).float()
                    start_out, end_out, phase_type_out, event_type_out = Model(data)
                    
                    # ploting(i,data=target['oridata'],start_pre=start_out.argmax(-1),end_pre=end_out.argmax(-1),start_label=starter_v,end_label=ender_v,type=[str(int(phase_typer_v))],pre_phase=int(phase_type_out.argmax(-1)))
                    #预测向量输出绘制
                    # df = pd.DataFrame(start_out[0,0,:].cpu(),columns=[str(typer_v)])
                    # df.plot(figsize=(16,8),)
                    # plt.savefig(f'./画图/概率输出/start{i}.png')
                    # plt.close()
                    
                    
                    # for i in range(4):
                    #     img=np.load(f'revise_4+1024+gausse/subfig1/{i*1024}_{i*1024+1024}.npy')
                    #     img=torch.from_numpy(img)
                    #     type_out,start_out,end_out=Model(img.unsqueeze(0))           
                    #     start_out=start_out.squeeze()
                    #     end_out=end_out.squeeze()
                    #     np.save(f'revise_4+1024+gausse/subfig1/result/{i*1024}_{i*1024+1024}_possibility_start.npy',start_out.cpu().numpy())
                    #     np.save(f'revise_4+1024+gausse/subfig1/result/{i*1024}_{i*1024+1024}_possibility_send.npy',end_out.cpu().numpy())
                    if int(phase_typer_v) == 0:
                        params.c_num += 1
                        if phase_type_out.argmax(-1) == int(phase_typer_v) and phase_type_out[0,0,int(phase_typer_v)] > 0.8:
                            np.save(f'论文插图数据/{int(phase_typer_v)}/{i}.npy', data.cpu().numpy())
                            params.phase_c += 1
                            if abs(start_out.argmax(-1) - starter_v) <= args.distance:
                                params.c_start += 1
                        elif phase_type_out.argmax(-1) != int(phase_typer_v):
                            params.phase_confused_metrix[0][int(phase_type_out.argmax(-1))] += 1
                            # ploting(i,data=target['oridata'],start_pre=start_out.argmax(-1),end_pre=end_out.argmax(-1),start_label=starter_v,end_label=ender_v,type=[str(phase_typer_v)])
                    elif int(phase_typer_v) == 1:
                        #ploting(i,data=target['oridata'],start_pre=start_out.argmax(-1),end_pre=end_out.argmax(-1),start_label=starter_v,end_label=ender_v,type=[str(phase_typer_v.item())],pre_phase=phase_type_out.argmax(-1))
                        params.p_num += 1
                        if phase_type_out.argmax(-1) == int(phase_typer_v) and phase_type_out[0,0,int(phase_typer_v)] > 0.8:
                            np.save(f'论文插图数据/{int(phase_typer_v)}/{i}.npy', data.cpu().numpy())
                            params.phase_p += 1
                            # ploting(i,data=target['oridata'],start_pre=start_out.argmax(-1),end_pre=end_out.argmax(-1),start_label=starter_v,end_label=ender_v,type=[str(phase_typer_v)],pre_phase=phase_type_out.argmax(-1))
                            # if abs(start_out.argmax(-1) - starter_v) <= args.distance:
                            #     params.p_start += 1
                            # if abs(end_out.argmax(-1) - ender_v) <= args.distance:
                            #     params.p_end += 1
                        elif phase_type_out.argmax(-1) != int(phase_typer_v) or phase_type_out[0,0,int(phase_typer_v)] < 0.8:
                            params.phase_confused_metrix[1][int(phase_type_out.argmax(-1))] += 1
                            # ploting(i,data=target['oridata'],start_pre=start_out.argmax(-1),end_pre=end_out.argmax(-1),start_label=starter_v,end_label=ender_v,type=[str(phase_typer_v)],pre_phase=phase_type_out.argmax(-1))
                        if event_typer_v == 0:
                            params.flare_num += 1
                            if phase_type_out.argmax(-1) == int(phase_typer_v) and phase_type_out[0,0,int(phase_typer_v)] > 0.8 and event_type_out[0,0,:].argmax(-1) == int(event_typer_v):
                                params.flare_id += 1                              
                        elif event_typer_v == 1:
                            params.micro_num += 1
                            if phase_type_out.argmax(-1) == int(phase_typer_v) and phase_type_out[0,0,int(phase_typer_v)] > 0.8 and event_type_out[0,0,:].argmax(-1) == int(event_typer_v):
                                params.micro_id += 1
                    elif int(phase_typer_v) == 2 :
                        params.g_num += 1
                        if phase_type_out.argmax(-1) == int(phase_typer_v) and phase_type_out[0,0,int(phase_typer_v)] > 0.8:
                            np.save(f'论文插图数据/{int(phase_typer_v)}/{i}.npy', data.cpu().numpy())
                            params.phase_g += 1
                            if abs(end_out.argmax(-1) - ender_v) <= args.distance:
                                params.g_end += 1    
                        elif phase_type_out.argmax(-1) != int(phase_typer_v):
                            params.phase_confused_metrix[2][int(phase_type_out.argmax(-1))] += 1
                            # ploting(i,data=target['oridata'],start_pre=start_out.argmax(-1),end_pre=end_out.argmax(-1),start_label=starter_v,end_label=ender_v,type=[str(phase_typer_v)],pre_phase=phase_type_out.argmax(-1))
                        # if phase_type_out.argmax(-1) == 3:
                        #     ploting(i,data=target['oridata'],start_pre=start_out.argmax(-1),end_pre=end_out.argmax(-1),start_label=starter_v,end_label=ender_v,type=[str(int(phase_typer_v))],pre_phase=int(phase_type_out.argmax(-1)))
                    elif int(phase_typer_v) == 3:
                        params.norm_num += 1
                        # ploting(i,data=target['oridata'],start_pre=start_out.argmax(-1),end_pre=end_out.argmax(-1),start_label=starter_v,end_label=ender_v,type=[str(phase_typer_v.item())],pre_phase=phase_type_out.argmax(-1))
                        if phase_type_out.argmax(-1) == int(phase_typer_v) and phase_type_out[0,0,int(phase_typer_v)] > 0.8:
                            np.save(f'论文插图数据/{int(phase_typer_v)}/{i}.npy', data.cpu().numpy())
                            params.norm_id += 1
                        elif phase_type_out.argmax(-1) != int(phase_typer_v):
                            params.phase_confused_metrix[3][int(phase_type_out.argmax(-1))] += 1

                    #ploting(i,data=target['oridata'],start_pre=start_out.argmax(-1),end_pre=end_out.argmax(-1),start_label=starter_v,end_label=ender_v,type=[str(phase_typer_v)])
                print(f'normrate:{round(params.norm_id/(params.norm_num+params.elps),2)}' + 
                f'||flarerate:{round((params.flare_id)/(params.flare_num+params.elps),2)}' +
                f'||microrate:{round(params.micro_id/(params.micro_num+params.elps),2)}' +
                f'||c_phase_rate:{round(params.phase_c/(params.c_num+params.elps),2)}' +
                f'||p_phase_rate:{round(params.phase_p/(params.p_num+params.elps),2)}' +
                f'||g_phase_rate:{round(params.phase_g/(params.g_num+params.elps),2)}'
                )
           
                print(f'||c_start_rate:{round((params.c_start)/(params.c_num+params.elps),2)}' +
                f'||g_end_rate:{round((params.g_end)/(params.g_num+params.elps),2)}' ) 
      
                rate={}   
                rate['normrate'] = round(params.norm_id/(params.norm_num+params.elps),2)
                rate['c_phase_rate'] = round(params.phase_c/(params.c_num+params.elps),2)
                rate['p_phase_rate'] = round(params.phase_p/(params.p_num+params.elps),2)
                rate['g_phase_rate'] = round(params.phase_g/(params.g_num+params.elps),2)

                rate['flarerate'] = round((params.flare_id)/(params.flare_num+params.elps),2)
                rate['microrate'] = round(params.micro_id/(params.micro_num+params.elps),2)
                rate['c_start_rate'] = round((params.c_start)/(params.c_num+params.elps),2)
                rate['g_end_rate'] = round((params.g_end)/(params.g_num+params.elps),2)

                return rate
    # weightlist=os.listdir('revise_4+1024+gausse/weight')
    if args.mode !='train':
        for weight in range(1):
            validation(args.weight)

    error = 0
    if args.mode=='train':
        # checkpoint=torch.load('AlertE_cancel_coding/65_0.pth',map_location=device)
        # Model.load_state_dict(checkpoint['m'])
        Model.train()
        os.makedirs(project_root+'weight',exist_ok=True)
        dataset=Mydata(dataset_dir=args.traindir,label_dir=args.labeldir,window=args.window)
        train_dataloader=DataLoader(dataset=dataset,batch_size=args.batchsize,shuffle=True,num_workers=16,pin_memory=True,drop_last=True)

        for epoch in range(args.epochs):
            loss_epoch = 0.0 
            loss_lr = 0.0
            for i,items in tqdm(enumerate(train_dataloader)):
                # try:
                if epoch == 0 and i == 0:
                    step = 0
                    loader_length = len(train_dataloader)
                # save_i = [0,loader_length//5,(loader_length//5)*2,(loader_length//5)*4]
                save_i = [0]
                data,target = items
                starter, ender,  phase_typer, event_typer= target['start'].to(device), target['end'].to(device), target['phase_type'].to(device), target['event_type'].to(device)
                data = data.to(device).float()
                start_out, end_out, phase_type_out, event_type_out = Model(data)
                optimizer.zero_grad()
                loss_phase_cls = cl_CELoss(phase_type_out.squeeze(1),phase_typer)
                start_lcoe = get_loss_coefficient (phase_typer, 'start')
                end_lcoe = get_loss_coefficient(phase_typer, 'end')
          
                loss_event_cls_resp = cl_CELoss1(event_type_out.squeeze(1),event_typer)
                event_lcoe = get_event_loss_coefficient(phase_typer)
                loss_event_cls = (loss_event_cls_resp * event_lcoe).mean(-1)
                
                loss_start_resp = lum_start_CELoss(start_out.squeeze(1),starter)
                loss_end_resp = lum_end_CELoss(end_out.squeeze(1),ender)
                loss_start = (start_lcoe * loss_start_resp).mean(-1)
                loss_end = (end_lcoe * loss_end_resp).mean(-1)
                
                loss = loss_phase_cls 
                # loss = loss_phase_cls
                loss_epoch += float(loss.item())
                # except RuntimeError:
                #     print(target['src'])
                #     continue
                if i == loader_length-1:
                    loss_lr = loss_epoch/loader_length
           
                loss.backward() 
                optimizer.step()
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar(project_root+"lr", current_lr, epoch*loader_length + i) 
                if i in save_i and epoch%1 == 0 and epoch >1:       
                    checkpoint = {
                        'epoch': epoch,
                        'm': Model.state_dict(),
                        'o': optimizer.state_dict()
                    }
                    torch.save(checkpoint,project_root+f"weight/{epoch}_{i}.pth")
                    step += 1
                    weight=project_root+f"weight/{epoch}_{i}.pth"
                    rate=validation(weight)
                    writer.add_scalar('rate/normrate',rate['normrate'],step)
                    writer.add_scalar('rate/c_phase',rate['c_phase_rate'],step)
                    writer.add_scalar('rate/p_phase',rate['p_phase_rate'],step)
                    writer.add_scalar('rate/g_phase',rate['g_phase_rate'],step)
                    # writer.add_scalar('rate/flarerate',rate['flarerate'],step)
                    # writer.add_scalar('rate/microrate',rate['microrate'],step)
                    # writer.add_scalar('rate/c_start_rate',rate['c_start_rate'],step)
                    # writer.add_scalar('rate/g_end_rate',rate['g_end_rate'],step)

                writer.add_scalar(project_root+'phase_type',loss_phase_cls.item(),i+epoch*loader_length)
                writer.add_scalar(project_root+'event_type',loss_event_cls.item(),i+epoch*loader_length)
                writer.add_scalar(project_root+'start',loss_start.item(),i+epoch*loader_length)
                writer.add_scalar(project_root+'end',loss_end.item(),i+epoch*loader_length)

                writer.add_scalar(project_root+'all',loss.item(),i+epoch*loader_length)
                writer.flush()
            #scheduler.step(loss_lr)
            




                



    