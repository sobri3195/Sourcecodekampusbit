package com.bagicode.myapplicationtensorflowjava.core;

import android.annotation.SuppressLint;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import com.bagicode.myapplicationtensorflowjava.entity.Classifier;
import com.bagicode.myapplicationtensorflowjava.entity.Recognition;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class TensorflowClassifier implements Recognition {

    private static final int MAX_RESULTS = 3;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;

    private Interpreter interpreter;
    private int inputSize;
    private List<String> labelList;

    private TensorflowClassifier() {}

    public static TensorflowClassifier init(AssetManager assetManager,
                                     String modelPath,
                                     String labelPath,
                                     int inputSize) throws IOException {

        TensorflowClassifier classifier = new TensorflowClassifier();
        classifier.interpreter = new Interpreter(FileUtils.loadModel(assetManager, modelPath));
        classifier.labelList = FileUtils.loadLabel(assetManager, labelPath);
        classifier.inputSize = inputSize;

        return classifier;
    }

    @Override
    public List<Classifier> recognize(Bitmap bitmap) {
        ByteBuffer byteBuffer = bmpToByteBuffer(bitmap);
        byte[][] result = new byte[1][labelList.size()];
        interpreter.run(byteBuffer, result);
        return sortPredicted(result);
    }

    @Override
    public void close() {
        interpreter.close();
        interpreter = null;
    }

    private ByteBuffer bmpToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.put((byte) ((val >> 16) & 0xFF));
                byteBuffer.put((byte) ((val >> 8) & 0xFF));
                byteBuffer.put((byte) (val & 0xFF));
            }
        }
        return byteBuffer;
    }

    @SuppressLint("DefaultLocale")
    private List<Classifier> sortPredicted(byte[][] labelProbArray) {

        PriorityQueue<Classifier> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Classifier>() {
                            @Override
                            public int compare(Classifier lhs, Classifier rhs) {
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int i = 0; i < labelList.size(); ++i) {
            float confidence = (labelProbArray[0][i] & 0xff) / 255.0f;
            if (confidence > THRESHOLD) {
                pq.add(new Classifier("" + i,
                        labelList.size() > i ? labelList.get(i) : "unknown",
                        confidence));
            }
        }

        final ArrayList<Classifier> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

}
