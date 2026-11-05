"""Assignment 2: reproducible CNN / frozen DINOv2 digit experiments."""
from pathlib import Path
import argparse, copy, hashlib, io, json, os, platform, random, time
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parent/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
import numpy as np
from PIL import Image
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'facebook/dinov2-small'
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(min(4, os.cpu_count() or 1))

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def read_dataset(root, out):
    """Use A1's existing split, never regenerate or move its files."""
    root, out = Path(root), Path(out)
    if (root/'split_dataset').is_dir(): root = root/'split_dataset'
    datasets, manifest, seen = {}, [], {}
    for split in ('train', 'val', 'test'):
        images, labels, names = [], [], []
        for label in range(10):
            files = sorted(p for p in (root/split/str(label)).glob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg'))
            if not files: raise ValueError(f'Missing images: {root/split/str(label)}')
            for p in files:
                with Image.open(p) as im:
                    im = im.convert('L')
                    if im.size != (28,28): raise ValueError(f'Expected processed 28x28 input: {p}')
                    arr = np.asarray(im).copy()
                digest = hashlib.sha256(arr.tobytes()).hexdigest()
                if digest in seen and seen[digest] != split:
                    raise ValueError(f'Cross-split duplicate: {p}; resolve leakage before training')
                seen[digest] = split
                name = p.relative_to(root).as_posix()
                manifest.append(dict(path=name, label=label, split=split, sha256=digest,
                                     source='own' if '_my_' in p.name else 'downloaded'))
                images.append(arr); labels.append(label); names.append(name)
        datasets[split] = (np.stack(images), np.array(labels, dtype=np.int64), names)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'dataset_manifest.json', manifest)
    np.savez_compressed(out/'dataset.npz', **{f'{s}_{v}': datasets[s][i] for s in datasets for i,v in enumerate(('images','labels'))})
    return datasets

def augment(image, rng, max_angle=10, max_shift=2, zoom=.08):
    matrix = cv2.getRotationMatrix2D((13.5,13.5), float(rng.uniform(-max_angle,max_angle)), float(rng.uniform(1-zoom,1+zoom)))
    matrix[:,2] += rng.uniform(-max_shift,max_shift,2)
    return cv2.warpAffine(image, matrix, (28,28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def prepare_upload(image, mode='photo'):
    """Photo: A1-style Otsu/crop/pad. Processed: preserve A1 28x28 pixels."""
    if image is None: raise ValueError('Choose an image or draw a digit first.')
    image = image.convert('RGBA')
    white = Image.new('RGBA', image.size, 'white'); white.alpha_composite(image)
    gray = np.asarray(white.convert('L'))
    if mode == 'processed':
        if gray.shape != (28,28): raise ValueError('Processed mode requires a 28 x 28 image with a bright digit on a dark background.')
        return gray.copy()
    if int(gray.max())-int(gray.min()) < 15: raise ValueError('The image is blank or has too little contrast.')
    border = np.concatenate((gray[0],gray[-1],gray[:,0],gray[:,-1]))
    threshold = cv2.THRESH_BINARY_INV if np.median(border)>127 else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray,0,255,threshold | cv2.THRESH_OTSU)
    contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError('No digit found. Use one clear digit on a plain background.')
    largest = max(contours,key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest)
    if w*h < 4: raise ValueError('Digit is too small.')
    side = max(w,h)+8
    canvas = np.zeros((side,side),dtype=np.uint8)
    dx,dy = (side-w)//2,(side-h)//2
    canvas[dy:dy+h,dx:dx+w] = binary[y:y+h,x:x+w]
    return cv2.resize(canvas,(28,28),interpolation=cv2.INTER_AREA)

class CNN(nn.Module):
    def __init__(self, improved=False):
        super().__init__()
        self.improved = improved
        if improved:
            self.net = nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout(.25),
                nn.Flatten(),nn.Linear(64*7*7,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,10))
        else:
            self.net = nn.Sequential(nn.Conv2d(1,16,3),nn.ReLU(),nn.MaxPool2d(2),
                nn.Flatten(),nn.Linear(16*13*13,32),nn.ReLU(),nn.Linear(32,10))
    def forward(self,x): return self.net(x)

def as_tensor(images):
    return torch.from_numpy(images.astype('float32')/255.)[:,None].to(DEVICE)

def dino_pixels(images):
    # Square digit images: resize without crop, keep the entire glyph.
    x = F.interpolate(as_tensor(images), size=(224,224),mode='bicubic',align_corners=False).clamp(0,1)
    x = x.repeat(1,3,1,1)
    mean = torch.tensor([.485,.456,.406],device=DEVICE)[None,:,None,None]
    std = torch.tensor([.229,.224,.225],device=DEVICE)[None,:,None,None]
    return (x-mean)/std

def load_backbone(out, download=True):
    local = Path(out)/'dinov2_backbone'
    if (local/'config.json').exists():
        model = AutoModel.from_pretrained(local,local_files_only=True)
    elif download:
        model = AutoModel.from_pretrained(MODEL_ID)
        model.save_pretrained(local, safe_serialization=True)
    else: raise FileNotFoundError('Run training first: DINOv2 weights are missing.')
    model.requires_grad_(False)
    return model.to(DEVICE).eval()

@torch.inference_mode()
def extract_features(model, images, batch_size=16, progress=True):
    chunks=[]
    for start in range(0,len(images),batch_size):
        chunks.append(model(pixel_values=dino_pixels(images[start:start+batch_size])).last_hidden_state[:,0].cpu())
        if progress and (start % 160 == 0 or start+batch_size>=len(images)):
            print(f'Features {min(start+batch_size,len(images))}/{len(images)}',flush=True)
    return torch.cat(chunks).numpy()

@torch.inference_mode()
def infer_cnn(model, images):
    model.eval()
    return torch.cat([model(as_tensor(images[i:i+64])).softmax(1).cpu() for i in range(0,len(images),64)]).numpy()

def train_cnn(data, out, improved=False):
    seed_all()
    model=CNN(improved).to(DEVICE)
    # Keras defaults use Glorot weights, zero biases, Adam eps=1e-7.
    for layer in model.modules():
        if isinstance(layer,(nn.Conv2d,nn.Linear)):
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-7)
    x,y,_=data['train']; vx,vy,_=data['val']
    history=[]; best=float('inf'); best_state=None; stale=0
    rng=np.random.default_rng(SEED)
    for epoch in range(30 if improved else 15):
        model.train(); total=correct=0; order=rng.permutation(len(x))
        for start in range(0,len(x),32):
            ids=order[start:start+32]; batch=x[ids]
            if improved: batch=np.stack([augment(im,rng,max_angle=28.8,max_shift=2.24) for im in batch])
            target=torch.from_numpy(y[ids]).to(DEVICE)
            optimizer.zero_grad(); logits=model(as_tensor(batch))
            loss=F.cross_entropy(logits,target); loss.backward(); optimizer.step()
            total+=loss.item()*len(ids); correct+=(logits.argmax(1)==target).sum().item()
        probs=infer_cnn(model,vx)
        val_loss=float(-np.log(np.clip(probs[np.arange(len(vy)),vy],1e-9,1)).mean())
        history.append(dict(epoch=epoch+1,loss=total/len(x),accuracy=correct/len(x),val_loss=val_loss,val_accuracy=float((probs.argmax(1)==vy).mean())))
        print(('Improved CNN' if improved else 'Baseline CNN'),history[-1],flush=True)
        if val_loss<best-1e-6: best=val_loss; best_state=copy.deepcopy(model.state_dict()); stale=0
        else: stale+=1
        if improved and stale>=5: break
    if improved: model.load_state_dict(best_state)
    name='cnn_improved' if improved else 'cnn_baseline'
    torch.save(model.cpu().state_dict(),Path(out)/f'{name}.pt')
    write_json(Path(out)/f'{name}_history.json',history)
    return model.to(DEVICE).eval(), history

def train_head(train, labels, val, val_labels, mean, std, out, name):
    seed_all()
    x=torch.from_numpy((train-mean)/std).float().to(DEVICE)
    y=torch.from_numpy(labels).to(DEVICE)
    vx=torch.from_numpy((val-mean)/std).float().to(DEVICE)
    vy=torch.from_numpy(val_labels).to(DEVICE)
    head=nn.Linear(train.shape[1],10).to(DEVICE)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.01)
    history=[]; best=float('inf'); best_state=None; stale=0
    # Full-batch updates keep the number of optimization steps comparable.
    for epoch in range(200):
        head.train(); optimizer.zero_grad(); logits=head(x)
        loss=F.cross_entropy(logits,y); loss.backward(); optimizer.step()
        head.eval()
        with torch.no_grad():
            v=head(vx); vl=F.cross_entropy(v,vy).item()
            row=dict(epoch=epoch+1,loss=loss.item(),accuracy=(logits.argmax(1)==y).float().mean().item(),val_loss=vl,val_accuracy=(v.argmax(1)==vy).float().mean().item())
        history.append(row)
        if vl<best-1e-6: best=vl; best_state=copy.deepcopy(head.state_dict()); stale=0
        else: stale+=1
        if epoch%20==0: print(name,row,flush=True)
        if stale>=20: break
    head.load_state_dict(best_state)
    torch.save(head.cpu().state_dict(),Path(out)/f'{name}.pt')
    write_json(Path(out)/f'{name}_history.json',history)
    return head.to(DEVICE).eval(),history

@torch.inference_mode()
def head_predict(head, features, mean, std):
    return head(torch.from_numpy((features-mean)/std).float().to(DEVICE)).softmax(1).cpu().numpy()

def metrics(y, probs):
    pred=probs.argmax(1)
    return dict(accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),
                loss=float(-np.log(np.clip(probs[np.arange(len(y)),y],1e-9,1)).mean()),
                report=classification_report(y,pred,labels=list(range(10)),output_dict=True,zero_division=0),
                confusion_matrix=confusion_matrix(y,pred,labels=list(range(10))).tolist())

def benchmark(fn, repetitions=30):
    for _ in range(5): fn()
    times=[]
    for _ in range(repetitions):
        if DEVICE.type=='cuda': torch.cuda.synchronize()
        t=time.perf_counter(); fn()
        if DEVICE.type=='cuda': torch.cuda.synchronize()
        times.append((time.perf_counter()-t)*1000)
    return dict(mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),n=repetitions,warmup=5,batch_size=1)

def run_experiments(dataset, out=ROOT/'artifacts', resume=False):
    seed_all(); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    old_manifest=json.loads((out/'dataset_manifest.json').read_text()) if (out/'dataset_manifest.json').exists() else None
    data=read_dataset(dataset,out)
    if resume and old_manifest != json.loads((out/'dataset_manifest.json').read_text()):
        raise ValueError('Cannot resume: dataset manifest changed.')
    print('Dataset:',{s:len(v[0]) for s,v in data.items()},flush=True)
    models={}; histories={}
    for improved in (False,True):
        key='cnn_improved' if improved else 'cnn_baseline'
        if resume and (out/f'{key}.pt').exists() and (out/f'{key}_history.json').exists():
            model=CNN(improved).to(DEVICE)
            model.load_state_dict(torch.load(out/f'{key}.pt',map_location=DEVICE,weights_only=True));model.eval()
            models[key]=model;histories[key]=json.loads((out/f'{key}_history.json').read_text())
            print('Resumed completed',key,flush=True)
        else: models[key],histories[key]=train_cnn(data,out,improved)
    backbone=load_backbone(out)
    features={s:extract_features(backbone,v[0]) for s,v in data.items() if s!='test'}
    rng=np.random.default_rng(SEED)
    augmented=np.stack([augment(im,rng) for _ in range(2) for im in data['train'][0]])
    np.savez_compressed(out/'augmented_training.npz',images=augmented,labels=np.tile(data['train'][1],2))
    augmented_features=extract_features(backbone,augmented)
    mean=features['train'].mean(0); std=features['train'].std(0).clip(.01)
    np.savez_compressed(out/'feature_bank.npz',features=features['train'],mean=mean,std=std,
                        images=data['train'][0],labels=data['train'][1])
    for name,tx,ty in [('dino_noaug',features['train'],data['train'][1]),
                       ('dino_aug',np.concatenate([features['train'],augmented_features]),np.tile(data['train'][1],3))]:
        models[name],histories[name]=train_head(tx,ty,features['val'],data['val'][1],mean,std,out,name)
    val_results={}
    for name,model in models.items():
        p=infer_cnn(model,data['val'][0]) if name.startswith('cnn') else head_predict(model,features['val'],mean,std)
        val_results[name]=metrics(data['val'][1],p)
    selected=min(('dino_noaug','dino_aug'),key=lambda name:val_results[name]['loss'])
    write_json(out/'selection.json',{'selected':selected,'criterion':'minimum validation cross-entropy between the two predefined DINOv2 heads; before test evaluation','validation':val_results})
    print('Selected using validation only:',selected,flush=True)
    # Test data is evaluated only after all model configurations are fixed.
    features['test']=extract_features(backbone,data['test'][0])
    results={}; predictions={}; latency={}
    for name,model in models.items():
        if name.startswith('cnn'):
            probs=infer_cnn(model,data['test'][0]); fn=lambda m=model:infer_cnn(m,data['test'][0][:1])
        else:
            probs=head_predict(model,features['test'],mean,std)
            fn=lambda m=model:head_predict(m,extract_features(backbone,data['test'][0][:1],progress=False),mean,std)
        results[name]=metrics(data['test'][1],probs); predictions[name]=probs.tolist()
        latency[name]=benchmark(fn)
        print(name,results[name]['accuracy'],results[name]['macro_f1'],latency[name],flush=True)
    write_json(out/'metrics.json',results); write_json(out/'predictions.json',predictions); write_json(out/'latency.json',latency)
    import transformers, sklearn
    provenance=json.loads((out/'dinov2_backbone/download_provenance.json').read_text()) if (out/'dinov2_backbone/download_provenance.json').exists() else {}
    write_json(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),device=str(DEVICE),cpu_threads=torch.get_num_threads(),torch=torch.__version__,transformers=transformers.__version__,sklearn=sklearn.__version__,numpy=np.__version__,seed=SEED,backbone=MODEL_ID,backbone_commit=provenance.get('revision',getattr(backbone.config,'_commit_hash',None)),timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z')))
    write_json(out/'experiment_config.json',dict(augmentation={'method':'affine image transformations, not generative AI','rotation_degrees':10,'translation_pixels':2,'zoom_fraction':.08,'copies_per_training_original':2,'generated_train_images':len(augmented),'validation_test_augmentation':False},head={'epochs_max':200,'patience':20,'optimizer':'AdamW','learning_rate':.01,'weight_decay':.01,'batch':'full'},synthetic_generator_share=0,selected=selected))
    print('COMPLETE:',out,flush=True)
    return results

class Predictor:
    def __init__(self, artifacts=ROOT/'artifacts'):
        self.out=Path(artifacts); self.backbone=load_backbone(self.out,False)
        self.bank=np.load(self.out/'feature_bank.npz')
        self.mean,self.std=self.bank['mean'],self.bank['std']
        self.selected=json.loads((self.out/'selection.json').read_text())['selected']
        self.head=nn.Linear(384,10).to(DEVICE)
        self.head.load_state_dict(torch.load(self.out/f'{self.selected}.pt',map_location=DEVICE,weights_only=True)); self.head.eval()
        f=self.bank['features']; self.normalized=f/np.linalg.norm(f,axis=1,keepdims=True).clip(1e-8)
        self.cnn=CNN().to(DEVICE)
        self.cnn.load_state_dict(torch.load(self.out/'cnn_baseline.pt',map_location=DEVICE,weights_only=True)); self.cnn.eval()
    def predict(self, pixels, model='dino'):
        if DEVICE.type=='cuda': torch.cuda.synchronize()
        start=time.perf_counter()
        if model=='baseline':
            probs=infer_cnn(self.cnn,pixels[None])[0]
            embedding=None
        else:
            embedding=extract_features(self.backbone,pixels[None],progress=False)
            probs=head_predict(self.head,embedding,self.mean,self.std)[0]
        if DEVICE.type=='cuda': torch.cuda.synchronize()
        inference_ms=(time.perf_counter()-start)*1000
        # Retrieval is separate from the timed prediction. Training images only.
        if embedding is None: embedding=extract_features(self.backbone,pixels[None],progress=False)
        query=embedding[0]/max(float(np.linalg.norm(embedding[0])),1e-8)
        similarities=self.normalized@query; neighbors=np.argsort(-similarities)[:5]
        return probs,neighbors,similarities[neighbors],inference_ms

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data',required=True); parser.add_argument('--out',default=str(ROOT/'artifacts'))
    parser.add_argument('--resume',action='store_true',help='Reuse completed CNN weights after checking the exact dataset manifest; use only with unchanged training code.')
    args=parser.parse_args(); run_experiments(args.data,args.out,args.resume)
