/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const IMAGE_SIZE = 32*32;  // 图片像素大小
const IMAGE_CHANNELS = 3;  // 图片通道数
const NUM_CLASSES = 10;  // 数据集类别数
const NUM_DATASET_ELEMENTS = 60000;  // 数据集样本数

const NUM_TRAIN_ELEMENTS = 50000;  // 训练集数量
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;  // 测试集数量
// 联网数据
// const IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/cifar10_images.png';
// const LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/cifar10_labels_uint8';
// 本地数据
const IMAGES_SPRITE_PATH = 'data/cifar10_images.png';  // 本地图像文件路径
const LABELS_PATH = 'data/cifar10_labels_uint8';  // 本地图像标签路径

export class Data {
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;

    }

    async load() {
        // Make a request for the MNIST sprited image.
        const img = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const imgRequest = new Promise((resolve, reject) => {
            img.crossOrigin = '';
            img.onload = () => {
                img.width = img.naturalWidth;
                img.height = img.naturalHeight;

                const datasetBytesBuffer =
                    new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * IMAGE_CHANNELS * 4 );
                const chunkSize = 5000;
                canvas.width = img.width;
                canvas.height = chunkSize;

                for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
                    const datasetBytesView = new Float32Array(
                        datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * IMAGE_CHANNELS* 4 ,
                        IMAGE_SIZE * chunkSize * IMAGE_CHANNELS);
                    ctx.drawImage(
                        img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
                        chunkSize);

                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

                    for (let j = 0; j < imageData.data.length / 4; j++) {
                        // All channels hold an equal value since the image is grayscale, so
                        // just read the red channel.
                        for (let k = 0; k < IMAGE_CHANNELS; k++) {
                            datasetBytesView[j * IMAGE_CHANNELS + k] = imageData.data[j * 4 + k] / 255;
                        }
                        // datasetBytesView[j] = imageData.data[j * 4] / 255;
                    }
                }
                this.datasetImages = new Float32Array(datasetBytesBuffer);
                // console.log(this.datasetImages);
                resolve();
            };
            img.src = IMAGES_SPRITE_PATH;
        });

        const labelsRequest = fetch(LABELS_PATH);
        const [imgResponse, labelsResponse] =
            await Promise.all([imgRequest, labelsRequest]);


        this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

        // Create shuffled indices into the train/test set for when we select a
        // random dataset element for training / validation.
        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

        // Slice the the images and labels into train and test sets.
        this.trainImages =
            this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS * IMAGE_CHANNELS);
        this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS * IMAGE_CHANNELS);
        this.trainLabels =
            this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
        this.testLabels =
            this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    }

    nextTrainBatch(batchSize) {
        return this.nextBatch(
            batchSize, [this.trainImages, this.trainLabels], () => {
                this.shuffledTrainIndex =
                    (this.shuffledTrainIndex + 1) % this.trainIndices.length;
                return this.trainIndices[this.shuffledTrainIndex];
            });
    }

    nextTestBatch(batchSize) {
        return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
            this.shuffledTestIndex =
                (this.shuffledTestIndex + 1) % this.testIndices.length;
            return this.testIndices[this.shuffledTestIndex];
        });
    }

    nextBatch(batchSize, data, index) {
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE * IMAGE_CHANNELS);
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

        for (let i = 0; i < batchSize; i++) {
            const idx = index();

            const image =
                data[0].slice(idx * IMAGE_SIZE * IMAGE_CHANNELS, IMAGE_CHANNELS * (idx * IMAGE_SIZE + IMAGE_SIZE));
            batchImagesArray.set(image, i * IMAGE_SIZE * IMAGE_CHANNELS);

            const label =
                data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
            batchLabelsArray.set(label, i * NUM_CLASSES);
        }

        const xs = tf.tensor3d(batchImagesArray, [batchSize, IMAGE_SIZE, IMAGE_CHANNELS]);
        const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

        return {xs, labels};
    }
}