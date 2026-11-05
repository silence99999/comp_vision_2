"""Integration checks against saved predictions and invalid image input."""
import io,json
import numpy as np
from PIL import Image
from app import app
from digit_pipeline import ROOT,prepare_upload

def image_file(array):
    stream=io.BytesIO();Image.fromarray(array).save(stream,format='PNG');stream.seek(0);return stream

def run_checks():
    app.config['TESTING']=True
    client=app.test_client()
    assert client.get('/').status_code==200
    info=client.get('/api/info');assert info.status_code==200
    examples=client.get('/api/examples');assert len(examples.json)==10
    data=np.load(ROOT/'artifacts/dataset.npz');preds=json.loads((ROOT/'artifacts/predictions.json').read_text())
    selected=info.json['selected']
    results=[]
    for index in (0,49,90):
        for model,key in [('dino',selected),('baseline','cnn_baseline')]:
            response=client.post('/api/predict',data={'image':(image_file(data['test_images'][index]),'digit.png'),'mode':'processed','model':model},content_type='multipart/form-data')
            assert response.status_code==200,response.json
            np.testing.assert_allclose(response.json['probabilities'],preds[key][index],rtol=2e-4,atol=2e-5)
            assert len(response.json['neighbors'])==5
            assert response.json['inference_ms']>0
            results.append({'index':index,'model':model,'digit':response.json['digit'],'matches_saved_probabilities':True})
    for pixels in (np.full((224,224),255,np.uint8),np.zeros((224,224),np.uint8)):
        response=client.post('/api/predict',data={'image':(image_file(pixels),'blank.png'),'mode':'photo','model':'dino'},content_type='multipart/form-data')
        assert response.status_code==400,response.json
    response=client.post('/api/predict',data={'image':(io.BytesIO(b'not an image'),'bad.png')},content_type='multipart/form-data')
    assert response.status_code==400
    assert client.post('/api/predict',data={}).status_code==400
    pixels=data['test_images'][0];assert np.array_equal(prepare_upload(Image.fromarray(pixels),'processed'),pixels)
    report={'status':'passed','checks':['home route','experiment info','10 test examples','six predictions agree with offline evaluation','five nearest neighbors','positive latency','white and black blank photo rejection','invalid image rejection','missing file rejection','processed mode preserves pixels'],'predictions':results}
    (ROOT/'artifacts/app_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':run_checks()
