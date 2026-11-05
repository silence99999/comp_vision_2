"""Create figures from saved experimental artifacts, never invented metrics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from digit_pipeline import ROOT

NAMES={'cnn_baseline':'Baseline CNN rerun','cnn_improved':'Improved CNN rerun','dino_noaug':'DINOv2 without augmentation','dino_aug':'DINOv2 with augmentation'}
COLORS=['#657b83','#c37b32','#4589a1','#14685a']

def create_figures(root=ROOT):
    root=Path(root); a=root/'artifacts'; dest=root/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    metrics=json.loads((a/'metrics.json').read_text()); predictions=json.loads((a/'predictions.json').read_text())
    selected=json.loads((a/'selection.json').read_text())['selected']
    data=np.load(a/'dataset.npz'); bank=np.load(a/'feature_bank.npz')
    fig,axes=plt.subplots(2,10,figsize=(10,2.6))
    manifest=json.loads((a/'dataset_manifest.json').read_text()); train=[v for v in manifest if v['split']=='train']
    for col in range(10):
        for row,source in enumerate(('own','downloaded')):
            idx=next(i for i,v in enumerate(train) if v['label']==col and v['source']==source)
            axes[row,col].imshow(data['train_images'][idx],cmap='gray',vmin=0,vmax=255);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0: axes[row,col].set_title(str(col))
            if col==0: axes[row,col].set_ylabel(source.title())
    fig.tight_layout();fig.savefig(dest/'dataset_examples.png',bbox_inches='tight');plt.close(fig)
    aug=np.load(a/'augmented_training.npz')
    fig,axes=plt.subplots(2,6,figsize=(9,3))
    for j,ax in enumerate(axes.flat):
        idx=(j*67)%len(aug['images']);ax.imshow(aug['images'][idx],cmap='gray',vmin=0,vmax=255);ax.set_title(f"Affine variant · {aug['labels'][idx]}",fontsize=9);ax.axis('off')
    fig.tight_layout();fig.savefig(dest/'augmentation_examples.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6.2))
    for ax,(key,title),color in zip(axes.flat,NAMES.items(),COLORS):
        h=json.loads((a/f'{key}_history.json').read_text());epochs=[v['epoch'] for v in h]
        ax.plot(epochs,[v['accuracy'] for v in h],color=color,label='Train');ax.plot(epochs,[v['val_accuracy'] for v in h],color=color,linestyle='--',label='Validation');ax.set_title(title,fontsize=10);ax.set_xlabel('Epoch');ax.set_ylabel('Accuracy');ax.set_ylim(0,1.03);ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(dest/'accuracy_curves.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6.2))
    for ax,(key,title),color in zip(axes.flat,NAMES.items(),COLORS):
        h=json.loads((a/f'{key}_history.json').read_text());epochs=[v['epoch'] for v in h]
        ax.plot(epochs,[v['loss'] for v in h],color=color,label='Train');ax.plot(epochs,[v['val_loss'] for v in h],color=color,linestyle='--',label='Validation');ax.set_title(title,fontsize=10);ax.set_xlabel('Epoch');ax.set_ylabel('Cross-entropy');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(dest/'loss_curves.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,key in zip(axes,('cnn_baseline',selected)):
        cm=np.array(metrics[key]['confusion_matrix']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=10)
        for i in range(10):
            for j in range(10): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,color='white' if cm[i,j]>5 else 'black')
        ax.set_xticks(range(10));ax.set_yticks(range(10));ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(NAMES[key],fontsize=10)
    fig.tight_layout();fig.savefig(dest/'confusion_matrices.png',bbox_inches='tight');plt.close(fig)
    # Actual queries and nearest-neighbor retrieval using saved model features.
    from digit_pipeline import Predictor
    predictor=Predictor(a)
    prob=np.array(predictions[selected]);wrong=np.flatnonzero(prob.argmax(1)!=data['test_labels'])
    ids=list(wrong[:2])
    if len(ids)<2: ids += [i for i in [0,90] if i not in ids][:2-len(ids)]
    fig,axes=plt.subplots(2,6,figsize=(10,3.7))
    examples=[]
    for row,idx in enumerate(ids):
        probs,near,sims,ms=predictor.predict(data['test_images'][idx]);ax=axes[row,0]
        ax.imshow(data['test_images'][idx],cmap='gray');ax.set_title(f'Query true {data["test_labels"][idx]}\npredicted {probs.argmax()}',fontsize=9);ax.axis('off')
        examples.append({'test_index':int(idx),'true':int(data['test_labels'][idx]),'predicted':int(probs.argmax()),'neighbors':[int(bank['labels'][j]) for j in near]})
        for ax,j,s in zip(axes[row,1:],near,sims):
            ax.imshow(bank['images'][j],cmap='gray');ax.set_title(f'Train label {bank["labels"][j]}\ncosine {s:.3f}',fontsize=9);ax.axis('off')
    fig.tight_layout(h_pad=3.0);fig.savefig(dest/'nearest_neighbors.png',bbox_inches='tight');plt.close(fig)
    (a/'explanation_examples.json').write_text(json.dumps(examples,indent=2))
    fig,ax=plt.subplots(figsize=(10,2.6));ax.set_xlim(0,10);ax.set_ylim(0,2.6);ax.axis('off')
    boxes=[(.05,1.15,1.6,.9,'A1 digit\n28 × 28'),(2.05,1.15,1.65,.9,'Resize 224 × 224\nRGB normalize'),(4.1,1.15,1.7,.9,'Frozen DINOv2\n384 features'),(6.25,1.15,1.65,.9,'Train linear head\n10 classes'),(8.35,1.15,1.6,.9,'Prediction\n0 to 9')]
    for x,y,w,h,label in boxes:ax.add_patch(Rectangle((x,y),w,h,facecolor='#eef4f1',edgecolor='#77998d'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
    for b,c in zip(boxes,boxes[1:]):ax.add_patch(FancyArrowPatch((b[0]+b[2],1.6),(c[0],1.6),arrowstyle='->',mutation_scale=13,color='#3d5c50'))
    ax.annotate('Training features → cosine similarity → nearest examples',xy=(4.95,1.15),xytext=(4.95,.35),ha='center',va='center',fontsize=10,arrowprops=dict(arrowstyle='<-',color='#3d5c50'))
    fig.tight_layout();fig.savefig(dest/'pipeline.png',bbox_inches='tight');plt.close(fig)
    # A faithful rendered results output figure, clearly named as such (not a browser screenshot).
    lines=['ACTUAL LOCAL EXPERIMENT OUTPUT','Model                         Accuracy    Macro F1', '-'*53]
    for key,m in metrics.items(): lines.append(f'{key:29s} {m["accuracy"]:8.4f}    {m["macro_f1"]:8.4f}')
    fig,ax=plt.subplots(figsize=(9,2.5));fig.patch.set_facecolor('#202626');ax.set_facecolor('#202626');ax.axis('off')
    ax.text(.03,.94,'\n'.join(lines),family='monospace',color='#e8eeee',va='top',fontsize=11)
    fig.savefig(dest/'results_output.png',bbox_inches='tight');plt.close(fig)
    print('Figures saved:',dest)

if __name__=='__main__': create_figures()
