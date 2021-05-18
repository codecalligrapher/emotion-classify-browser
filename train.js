// let numClasses = 7
// // training parameters
// let batchSize = 32
// let epochs = 100
// let image_shape = [48, 48, 1]
// let verbose = true 
// let num_class = 7
// let patience = 50  // number of epochs with no improvement after which training will be stopped
// let base_path = 'models/'
// let l2_reg = 0.01

// const reg = tf.regularizers.l2()
// // Model
// const input = tf.input({shape: [48, 48, 1]});
// let x = tf.layers.conv2d({filters: 8, kernelSize: [3,3], strides:[1,1], padding: 'same', kernelRegularizer: reg, useBias: false}).apply(input);
// x = tf.layers.batchNormalization().apply(x);
// x = tf.layers.activation({activation: 'relu'}).apply(x);
// x = tf.layers.conv2d({filters: 8, kernelSize: [3,3], strides:[1,1], padding: 'same', kernelRegularizer: reg, useBias: false}).apply(x);
// x = tf.layers.batchNormalization().apply(x);
// x = tf.layers.activation({activation: 'relu'}).apply(x);

// //module1: residual 


// let residual = tf.layers.conv2d({filters:16, kernelSize:(1,1), strides:(2,2), padding:'same', useBias:false}).apply(x)
// residual = tf.layers.batchNormalization().apply(residual)
 
// x = tf.layers.separableConv2d({filters:16 , kernelSize:(3,3), padding:'same'}).apply(x)
// x = tf.layers.batchNormalization().apply(x)
// x = tf.layers.activation({activation:'relu'}).apply(x)
// x = tf.layers.separableConv2d({filters:16, kernelSize:(3,3), padding:'same'}).apply(x)
// x = tf.layers.batchNormalization().apply(x)
// x = tf.layers.maxPooling2d({poolSize:(3,3), strides:(2,2), padding:'same'}).apply(x)
// x = tf.layers.add().apply([x,residual])


// // module 2
// // residual module 
// residual = tf.layers.conv2d({filters:32, kernelSize:(1,1), strides:(2,2), padding:'same', useBias:false}).apply(x)
// residual = tf.layers.batchNormalization().apply(residual)
// x = tf.layers.separableConv2d({filters:32, kernelSize:(3,3), padding:'same'}).apply(x)
// x = tf.layers.batchNormalization().apply(x)
// x = tf.layers.activation({activation:'relu'}).apply(x)
// x = tf.layers.separableConv2d({filters:32, kernelSize:(3,3), padding:'same'}).apply(x)
// x = tf.layers.batchNormalization().apply(x)
// x = tf.layers.maxPooling2d({pool_size:(3,3), strides:(2,2), padding:'same'}).apply(x)
// x = tf.layers.add().apply([x,residual])

// // module 3
// // residual module 
// residual = tf.layers.conv2d({filters:64, kernelSize:(1,1), strides:(2,2), padding:'same', useBias:false}).apply(x)
// residual = tf.layers.batchNormalization().apply(residual)

// x = tf.layers.separableConv2d({filters:64, kernelSize:(3,3), padding:'same', useBias:false}).apply(x)
// x = tf.layers.batchNormalization().apply(x)
// x = tf.layers.activation({activation:'relu'}).apply(x)
// x = tf.layers.separableConv2d({filters:64, kernelSize:(3,3), padding:'same',  useBias:false}).apply(x)
// x = tf.layers.batchNormalization().apply(x)
// x = tf.layers.maxPooling2d({pool_size:(3,3), strides:(2,2), padding:'same'}).apply(x)
// x = tf.layers.add().apply([x,residual])

// // module 4
// // residual module 
// residual = tf.layers.conv2d({filters:128, kernelSize:(1,1), strides:(2,2), padding:'same', useBias:false}).apply(x)
// residual = tf.layers.batchNormalization().apply(residual)

// x = tf.layers.separableConv2d({filters:128, kernelSize:(3,3), padding:'same', useBias:false}).apply(x)
// x = tf.layers.batchNormalization().apply(x)
// x = tf.layers.activation({activation:'relu'}).apply(x)
// x = tf.layers.separableConv2d({filters:128, kernelSize:(3,3), padding:'same', useBias:false}).apply(x)
// x = tf.layers.batchNormalization().apply(x)
// x = tf.layers.maxPooling2d({pool_size:(3,3), strides:(2,2), padding:'same'}).apply(x)
// x = tf.layers.add().apply([x,residual])

// x = tf.layers.conv2d({filters:numClasses, kernelSize:(3,3), padding:'same'}).apply(x)
// x = tf.layers.globalAveragePooling2d({dataFormat: 'channelsFirst'}).apply(x)

// const output = tf.layers.activation('softmax').apply(x)

// const model = tf.model({inputs: input, outputs: output});
// model.compile({
//     loss: 'categoricalCrossentropy',
//     optimizer: 'adam',
//     metrics: ['accuracy']
// })
// const  onTrainBegin = logs => {
//     console.log("onTrainBegin");
// }


// const history = await model.fit(xs, ys, {
//     epochs: epochs,
//     batchSize: batchSize,
//     callbacks: tf.callbacks.earlyStopping({monitor: 'val_acc'})
//  });

let model;
async function load() {
    model = await tf.loadLayersModel('xception/model.json');
    const img = document.getElementById('img');

    const tensor = tf.browser.fromPixels(img,1).expandDims(0)
    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = tensor.sub(offset).div(offset);
    const res =  model.predict(normalized)
    res.print()

 }

load();
