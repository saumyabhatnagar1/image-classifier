let net;
const classifier=knnClassifier.create()
const webcamElement=document.getElementById('webcam')
async function app() {
    console.log('Loading mobilenet..');
  
    net = await mobilenet.load();
    console.log('Successfully loaded model');
    const webcam=await tf.data.webcam(webcamElement)
  const addExample = async classId => {
    
    const img = await webcam.capture();

    const activation = net.infer(img, true);

    classifier.addExample(activation, classId);

    img.dispose();
  };

  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));

    while (true) {
      if(classifier.getNumClasses()>0)
      {
      const img=await webcam.capture()
      const activation=net.infer(img,'conv_preds')

      
      const result = await classifier.predictClass(activation);   
      const result2=await net.classify(img)   
      const classes=['A','B','C'];
        
      document.getElementById('console').innerText = `
        object:${result2[0].className}\n
      prediction: ${classes[result.label]}\n
      probability: ${result.confidences[result.label]}
    `;

      img.dispose();
      }
      await tf.nextFrame();
    }
  }
app()