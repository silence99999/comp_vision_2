"""Local-only web demonstration. Uploaded images are processed in memory."""
import base64, io, json, threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, UnidentifiedImageError
import numpy as np
from digit_pipeline import Predictor, prepare_upload, ROOT, DEVICE

app = Flask(__name__, static_folder='web')
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
predictor = None
lock = threading.Lock()
totals = {'dino': [0.,0], 'baseline': [0.,0]}

def png_uri(array):
    buff=io.BytesIO(); Image.fromarray(array).save(buff,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buff.getvalue()).decode()

def get_predictor():
    global predictor
    if predictor is None: predictor=Predictor()
    return predictor

@app.get('/')
def index(): return send_from_directory(ROOT/'web','index.html')

@app.get('/api/info')
def info():
    artifacts=ROOT/'artifacts'
    return jsonify(environment=json.loads((artifacts/'environment.json').read_text()),
                   metrics=json.loads((artifacts/'metrics.json').read_text()),
                   latency=json.loads((artifacts/'latency.json').read_text()),
                   selected=json.loads((artifacts/'selection.json').read_text())['selected'])

@app.get('/api/examples')
def examples():
    data=np.load(ROOT/'artifacts/dataset.npz')
    return jsonify([dict(label=d,image=png_uri(data['test_images'][np.flatnonzero(data['test_labels']==d)[0]])) for d in range(10)])

@app.post('/api/predict')
def predict():
    try:
        with lock:
            pred=get_predictor()
            mode=request.form.get('mode','photo'); model=request.form.get('model','dino')
            if mode not in ('photo','processed') or model not in ('dino','baseline'): raise ValueError('Invalid mode or model.')
            if 'image' not in request.files: raise ValueError('Choose an image first.')
            with Image.open(request.files['image'].stream) as im:
                if max(im.size)>4096: raise ValueError('Use an image no larger than 4096 pixels per side.')
                pixels=prepare_upload(im,mode)
            probs,indices,similarities,ms=pred.predict(pixels,model)
            totals[model][0]+=ms; totals[model][1]+=1
            return jsonify(digit=int(probs.argmax()),confidence=float(probs.max()),probabilities=probs.tolist(),
                processed=png_uri(pixels),inference_ms=ms,average_ms=totals[model][0]/totals[model][1],requests=totals[model][1],
                model=pred.selected if model=='dino' else 'cnn_baseline',device=str(DEVICE),
                neighbors=[dict(image=png_uri(pred.bank['images'][i]),label=int(pred.bank['labels'][i]),similarity=float(s)) for i,s in zip(indices,similarities)])
    except (ValueError,UnidentifiedImageError,OSError) as exc:
        return jsonify(error=str(exc)),400

@app.errorhandler(413)
def too_large(exc): return jsonify(error='Image is too large. Maximum upload is 8 MB.'),413

if __name__=='__main__':
    print('Loading trained models...')
    get_predictor()
    print('Open http://127.0.0.1:7860 — local only; Ctrl+C stops the app.')
    app.run(host='127.0.0.1',port=7860,debug=False,threaded=True)
